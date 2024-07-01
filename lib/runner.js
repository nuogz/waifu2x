import { readFileSync } from 'fs';

import { ImageData } from 'canvas';
import Onnx from 'onnxruntime-node';
import { resolve as resolvePath } from 'path';



/** @typedef {import('../bases.d.ts').ArchOption} ArchOption */
/** @typedef {import('../bases.d.ts').ScaleOption} ScaleOption */
/** @typedef {import('../bases.d.ts').TileOption} TileOption */

/** @typedef {import('../bases.d.ts').SeamBlendingParam} SeamBlendingParam */



/**
 * @param {ArchOption} arch
 * @param {ScaleOption} scale
 * @returns {number}
 */
export const calcOffset = (arch, scale) => scale * 8 + (arch == 'swin_unet' ? 0 : arch == 'cunet' ? 20 : NaN);

/**
 * @param {ArchOption} arch
 * @param {TileOption} tile
 * @param {number} [offset]
 * @returns {number}
 */
export const calcTileSize = (arch, tile, offset) => {
	if(arch == 'swin_unet') {
		while(true) {
			if(
				(tile - 16) % 12 == 0 &&
				(tile - 16) % 16 == 0
			) { break; }

			tile += 1;
		}

		return tile;
	}
	else if(arch == 'cunet') {
		tile = tile + (offset - 16) * 2;
		tile -= tile % 4;

		return tile;
	}

	throw Error('unknown arch');
};


/** @type {Object<string,Onnx.InferenceSession>} */
const sessionsOnnxUtil$name = {};
/**
 * @param {string} name
 * @param {string} dirModel
 * @returns {Promise<Onnx.InferenceSession>}
 */
export const getUtilOnnxSession = async (name, dirModel) => {
	if(name in sessionsOnnxUtil$name) { return sessionsOnnxUtil$name[name]; }

	return sessionsOnnxUtil$name[name] =
		await Onnx.InferenceSession.create(readFileSync(resolvePath(
			dirModel, 'utils', `${name}.onnx`
		)));
};


/**
 * @param {Uint8ClampedArray} rgba
 * @param {number} width
 * @param {number} height
 * @param {boolean} keepAlpha
 * @returns {[Onnx.TypedTensor<"float32">, Onnx.TypedTensor<"float32">, Onnx.TypedTensor<"float32">]}
 */
export const convertImageDataToTensor = (rgba, width, height, keepAlpha = false) => {
	if(keepAlpha) {
		const rgb = new Float32Array(height * width * 3);
		const alpha1 = new Float32Array(height * width * 1);
		const alpha3 = new Float32Array(height * width * 3);

		for(let y = 0; y < height; ++y) {
			for(let x = 0; x < width; ++x) {
				const i = (y * width * 4) + (x * 4);
				const j = (y * width + x);

				rgb[j] = rgba[i + 0] / 255.0;
				rgb[j + 1 * (height * width)] = rgba[i + 1] / 255.0;
				rgb[j + 2 * (height * width)] = rgba[i + 2] / 255.0;


				const alpha = rgba[i + 3] / 255.0;

				alpha1[j] = alpha;
				alpha3[j] = alpha;
				alpha3[j + 1 * (height * width)] = alpha;
				alpha3[j + 2 * (height * width)] = alpha;
			}
		}

		return [
			new Onnx.Tensor('float32', rgb, [1, 3, height, width]),
			new Onnx.Tensor('float32', alpha1, [1, 1, height, width]), // for mask
			new Onnx.Tensor('float32', alpha3, [1, 3, height, width]), // for upscaling with rgb input
		];
	}
	else {
		const rgb = new Float32Array(height * width * 3);
		const colorBackground = 1.0;

		for(let y = 0; y < height; ++y) {
			for(let x = 0; x < width; ++x) {
				let alpha = rgba[(y * width * 4) + (x * 4) + 3] / 255.0;

				for(let c = 0; c < 3; ++c) {
					let i = (y * width * 4) + (x * 4) + c;
					let j = (y * width + x) + c * (height * width);

					rgb[j] = alpha * (rgba[i] / 255.0) + (1 - alpha) * colorBackground;
				}
			}
		}

		return [
			new Onnx.Tensor('float32', rgb, [1, 3, height, width])
		];
	}
};



export const SeamBlending = class {
	static BLEND_SIZE = 4;


	/** @type {number[]} */
	sizesSource;
	/** @type {number} */
	scale;
	/** @type {number} */
	offset;
	/** @type {number} */
	tile;
	/** @type {number} */
	sizeBlend;

	/** @type {Onnx.Tensor} */
	filterBlend;

	/** @type {SeamBlendingParam} */
	param = {};


	/** @type {Onnx.TypedTensor<"float32">} */
	pixels;
	/** @type {Onnx.TypedTensor<"float32">} */
	weights;
	/** @type {Onnx.TypedTensor<"float32">} */
	output;


	/**
	 * Cumulative Tile Seam/Border Blending
	 * This function requires large buffers and does not work with onnxruntime's web-worker.
	 * So this function is implemented in non-async pure javascript.
	 * original code: nunif/utils/seam_blending.py
	 * @param {number[]} sizesSource
	 * @param {number} scale
	 * @param {number} offset
	 * @param {number} tile
	 * @param {number} sizeBlend
	 * @param {string} dirModel
	 */
	constructor(sizesSource, scale, offset, tile, sizeBlend = SeamBlending.BLEND_SIZE, dirModel) {
		this.sizesSource = sizesSource;
		this.scale = scale;
		this.offset = offset;
		this.tile = tile;
		this.sizeBlend = sizeBlend;


		this.calcParams();


		// NOTE: Float32Array is initialized by 0
		this.pixels = new Onnx.Tensor('float32',
			new Float32Array(this.param.heightBufferTarget * this.param.widthBufferTarget * 3),
			[3, this.param.heightBufferTarget, this.param.widthBufferTarget],
		);
		this.weights = new Onnx.Tensor('float32',
			new Float32Array(this.param.heightBufferTarget * this.param.widthBufferTarget * 3),
			[3, this.param.heightBufferTarget, this.param.widthBufferTarget],
		);



		return this.createSeamBlendingFilter(dirModel)
			.then(filterBlend => {
				this.filterBlend = filterBlend;

				this.output = new Onnx.Tensor('float32',
					new Float32Array(this.filterBlend.data.length),
					this.filterBlend.dims,
				);


				return Promise.resolve(this);
			});
	}

	// from nunif/utils/seam_blending.py
	calcParams() {
		const param = this.param;

		const heightSource = this.sizesSource[2];
		const widthSource = this.sizesSource[3];

		param.heightTarget = heightSource * this.scale;
		param.widthTarget = widthSource * this.scale;

		param.offsetInput = Math.ceil(this.offset / this.scale);
		param.sizeBlendInput = Math.ceil(this.sizeBlend / this.scale);
		param.stepTileInput = this.tile - (param.offsetInput * 2 + param.sizeBlendInput);
		param.stepTileOutput = param.stepTileInput * this.scale;


		let heightBlocks = 0;
		let widthBlocks = 0;
		let heightInput = 0;
		let widthInput = 0;
		while(heightInput < heightSource + param.offsetInput * 2) {
			heightInput = heightBlocks * param.stepTileInput + this.tile;

			heightBlocks++;
		}
		while(widthInput < widthSource + param.offsetInput * 2) {
			widthInput = widthBlocks * param.stepTileInput + this.tile;

			widthBlocks++;
		}


		param.heightBlocks = heightBlocks;
		param.widthBlocks = widthBlocks;
		param.heightBufferTarget = heightInput * this.scale;
		param.widthBufferTarget = widthInput * this.scale;

		param.pad = [
			param.offsetInput,
			widthInput - (widthSource + param.offsetInput),
			param.offsetInput,
			heightInput - (heightSource + param.offsetInput),
		];
	}


	/**
	 * @param {Onnx.Tensor} tensorSource
	 * @param {number} iTile
	 * @param {number} jTile
	 * @returns {Onnx.TypedTensor<"float32">}
	 */
	update(tensorSource, iTile, jTile) {
		const sizeStep = this.param.stepTileOutput;

		const [, H, W] = this.filterBlend.dims;

		const HW = H * W;

		const heightBuffer = this.pixels.dims[1];
		const widthBuffer = this.pixels.dims[2];
		const hwBuffer = heightBuffer * widthBuffer;

		const iHeight = sizeStep * iTile;
		const iWidth = sizeStep * jTile;


		let weightOld;
		let weightNext;
		let weightNew;
		for(let c = 0; c < 3; ++c) {
			for(let i = 0; i < H; ++i) {
				for(let j = 0; j < W; ++j) {
					let indexTile = c * HW + i * W + j;
					let indexBuffer = c * hwBuffer + (iHeight + i) * widthBuffer + (iWidth + j);

					weightOld = this.weights.data[indexBuffer];
					weightNext = weightOld + this.filterBlend.data[indexTile];
					weightOld = weightOld / weightNext;
					weightNew = 1.0 - weightOld;

					this.pixels.data[indexBuffer] = (this.pixels.data[indexBuffer] * weightOld + tensorSource.data[indexTile] * weightNew);
					this.weights.data[indexBuffer] += this.filterBlend.data[indexTile];
					this.output.data[indexTile] = this.pixels.data[indexBuffer];
				}
			}
		}


		return this.output;
	}


	/**
	 * @param {string} dirModel
	 * @returns {Onnx.Tensor}
	 */
	async createSeamBlendingFilter(dirModel) {
		const session = await getUtilOnnxSession('create_seam_blending_filter', dirModel);

		return (await session.run({
			scale: new Onnx.Tensor('int64', BigInt64Array.from([BigInt(this.scale)]), []),
			offset: new Onnx.Tensor('int64', BigInt64Array.from([BigInt(this.offset)]), []),
			tile_size: new Onnx.Tensor('int64', BigInt64Array.from([BigInt(this.tile)]), []),
		})).y;
	}
};



/**
 * @param {Onnx.Tensor} tensorColor
 * @param {Onnx.Tensor} tensorAlpha
 * @param {bigint} offset
 * @param {string} dirModel
 * @returns {Promise<Onnx.TypedTensor<"float32">>}
 */
export const padAlphaBorder = async (tensorColor, tensorAlpha, offset, dirModel) => {
	const session = await getUtilOnnxSession('alpha_border_padding', dirModel);

	// unsqueeze
	tensorColor = new Onnx.Tensor('float32', tensorColor.data, [tensorColor.dims[1], tensorColor.dims[2], tensorColor.dims[3]]);
	tensorAlpha = new Onnx.Tensor('float32', tensorAlpha.data, [tensorAlpha.dims[1], tensorAlpha.dims[2], tensorAlpha.dims[3]]);
	offset = new Onnx.Tensor('int64', BigInt64Array.from([offset]), []);

	const { y: tensorTarget } = await session.run({ rgb: tensorColor, alpha: tensorAlpha, offset });


	// squeeze
	return new Onnx.Tensor('float32', tensorTarget.data, [1, tensorTarget.dims[0], tensorTarget.dims[1], tensorTarget.dims[2]]);
};
/**
 * @param {Onnx.Tensor} tensorSource
 * @param {BigInt} left
 * @param {BigInt} right
 * @param {BigInt} top
 * @param {BigInt} bottom
 * @param {string} dirModel
 * @returns {Promise<Onnx.Tensor>}
 */
export const padding = async (tensorSource, left, right, top, bottom, dirModel) => {
	const session = await getUtilOnnxSession('pad', dirModel);

	return (await session.run({
		x: tensorSource,
		left: new Onnx.Tensor('int64', BigInt64Array.from([left]), []),
		right: new Onnx.Tensor('int64', BigInt64Array.from([right]), []),
		top: new Onnx.Tensor('int64', BigInt64Array.from([top]), []),
		bottom: new Onnx.Tensor('int64', BigInt64Array.from([bottom]), []),
	})).y;
};


/**
 * @param {Float32Array} z
 * @param {Float32Array} alpha3
 * @param {number} width
 * @param {number} height
 * @returns {ImageData}
 */
export const toImageData = (z, alpha3, width, height) => {
	// CHW -> HWC  0.0-1.0 -> 0-255
	const rgba = new Uint8ClampedArray(height * width * 4);

	if(alpha3 != null) {
		for(let y = 0; y < height; ++y) {
			for(let x = 0; x < width; ++x) {
				let vAlpha = 0.0;

				for(let c = 0; c < 3; ++c) {
					let i = (y * width * 4) + (x * 4) + c;
					let j = (y * width + x) + c * (height * width);

					rgba[i] = (z[j] * 255.0) + 0.49999;
					vAlpha += alpha3[j] * (1.0 / 3.0);
				}

				rgba[(y * width * 4) + (x * 4) + 3] = (vAlpha * 255.0) + 0.49999;
			}
		}
	}
	else {
		rgba.fill(255);

		for(let y = 0; y < height; ++y) {
			for(let x = 0; x < width; ++x) {
				for(let c = 0; c < 3; ++c) {
					let i = (y * width * 4) + (x * 4) + c;
					let j = (y * width + x) + c * (height * width);

					rgba[i] = (z[j] * 255.0) + 0.49999;
				}
			}
		}
	}


	return new ImageData(rgba, width, height);
};

/** @param {any[]} array */
export const shuffleArray = array => {
	for(let i = array.length - 1; i > 0; i--) {
		const j = Math.floor(Math.random() * (i + 1));

		[array[i], array[j]] = [array[j], array[i]];
	}
};


/**
 * @param {Uint8ClampedArray} rgba
 * @param {boolean} [keepAlpha=false]
 * @returns {[number, number, number, number]}
 */
export const checkSingleColor = (rgba, keepAlpha = false) => {
	let r = rgba[0];
	let g = rgba[1];
	let b = rgba[2];
	let a = rgba[3];


	for(let i = 0; i < rgba.length; i += 4) {
		if(r != rgba[i + 0] || g != rgba[i + 1] || b != rgba[i + 2] || a != rgba[i + 3]) {
			return null;
		}
	}

	if(keepAlpha) {
		return [r / 255.0, g / 255.0, b / 255.0, a / 255.0];
	}
	else {
		const colorBackground = 1.0;

		a = a / 255.0;
		r = a * (r / 255.0) + (1 - a) * colorBackground;
		g = a * (g / 255.0) + (1 - a) * colorBackground;
		b = a * (b / 255.0) + (1 - a) * colorBackground;

		return [r, g, b, 1.0];
	}
};


/**
 *
 * @param {Onnx.Tensor} tensorSource
 * @param {number} tta
 * @param {string} dirModel
 * @returns {Promise<Onnx.Tensor>}
 */
export const splitTTA = async (tensorSource, tta, dirModel) => {
	const session = await getUtilOnnxSession('tta_split', dirModel);

	return (await session.run({
		x: tensorSource,
		tta_level: new Onnx.Tensor('int64', BigInt64Array.from([tta]), [])
	})).y;
};
/**
 *
 * @param {Onnx.Tensor} tensorSource
 * @param {number} tta
 * @param {string} dirModel
 * @returns {Promise<Onnx.Tensor>}
 */
export const mergeTTA = async (tensorSource, tta, dirModel) => {
	const session = await getUtilOnnxSession('tta_merge', dirModel);

	return (await session.run({
		x: tensorSource,
		tta_level: new Onnx.Tensor('int64', BigInt64Array.from([tta]), [])
	})).y;
};


/**
 * @param {number[]} rgba
 * @param {number} size
 * @returns {[Onnx.TypedTensor<"float32">, Onnx.TypedTensor<"float32">]}
 */
export const createSingleColorTensor = (rgba, size) => {
	// CHW
	const color = new Float32Array(size * size * 3);
	const alpha3 = new Float32Array(size * size * 3);
	alpha3.fill(rgba[3]);

	for(let c = 0; c < 3; ++c) {
		const v = rgba[c];
		for(let i = 0; i < size * size; ++i) {
			color[c * size * size + i] = v;
		}
	}

	return [
		new Onnx.Tensor('float32', color, [1, 3, size, size]),
		new Onnx.Tensor('float32', alpha3, [1, 3, size, size])
	];
};








/**
 * @param {Uint8ClampedArray} rgba
 * @returns {boolean}
 */
export const checkAlphaChannel = rgba => {
	for(let i = 0; i < rgba.length; i += 4) {
		const alpha = rgba[i + 3];

		if(alpha != 255) { return true; }
	}

	return false;
};
