import { readFileSync } from 'fs';
import { dirname, resolve as resolvePath } from 'path';
import { fileURLToPath } from 'url';

import { createCanvas, loadImage } from 'canvas';
import Onnx from 'onnxruntime-node';

import {
	padAlphaBorder, calcOffset, calcTileSize, checkAlphaChannel, checkSingleColor,
	createSingleColorTensor, padding, SeamBlending, shuffleArray, toImageData, convertImageDataToTensor, mergeTTA, splitTTA
} from './lib/runner.js';



/** @typedef {import('./bases.d.ts').Waifu2xOption} Waifu2xOption */
/** @typedef {import('./bases.d.ts').ArchOption} ArchOption */
/** @typedef {import('./bases.d.ts').ScaleOption} ScaleOption */
/** @typedef {import('./bases.d.ts').Controller} Controller */
/** @typedef {import('./bases.d.ts').BlockFinishedHandle} BlockFinishedHandle */


const dirPackage = dirname(fileURLToPath(import.meta.url));



/**
 * @param {Buffer} buffer
 * @param {Waifu2xOption} option
 * @param {BlockFinishedHandle} emitBlockFinished
 * @param {Controller} controller
 * @returns {Promise<Buffer>}
 */
export default async function waifu2x(buffer, option, emitBlockFinished = () => { }, controller = { break: false }) {
	const { arch, style, noise, scale, tile, isTileShuffle, tta, enableAlphaChannel, dirModel: dirModel = resolvePath(dirPackage, 'model') } = option;


	const imageRaw = await loadImage(buffer);
	const canvasRaw = createCanvas(imageRaw.naturalWidth, imageRaw.naturalHeight);
	const contextRaw = canvasRaw.getContext('2d', { willReadFrequently: true });
	contextRaw.drawImage(imageRaw, 0, 0);
	const dataImageRaw = contextRaw.getImageData(0, 0, canvasRaw.width, canvasRaw.height);
	const hasAlpha = enableAlphaChannel ? checkAlphaChannel(dataImageRaw.data) : false;


	const model = await Onnx.InferenceSession.create(readFileSync(resolvePath(
		dirModel, arch, style, `${~noise ? `noise${noise}_` : ''}scale${scale}x.onnx`)
	));
	const modelAlpha = !hasAlpha ? null : await Onnx.InferenceSession.create(readFileSync(resolvePath(
		dirModel, arch, style, `scale${scale}x.onnx`)
	));


	const offset = calcOffset(arch, scale);
	const sizeTile = calcTileSize(arch, tile, offset);


	const [tensorColor, tensorAlpha, tensorAlpha3] = convertImageDataToTensor(dataImageRaw.data, dataImageRaw.width, dataImageRaw.height, hasAlpha);

	const seamBlendingColor = await new SeamBlending(tensorColor.dims, scale, offset, sizeTile, undefined, dirModel);
	const seamBlendingAlpha = hasAlpha ? await new SeamBlending(tensorAlpha3.dims, scale, offset, sizeTile, undefined, dirModel) : null;

	const paramsRender = seamBlendingColor.param;


	const tensorColorPadded = await padding(
		hasAlpha ? await padAlphaBorder(tensorColor, tensorAlpha, BigInt(offset), dirModel) : tensorColor,
		BigInt(paramsRender.pad[0]), BigInt(paramsRender.pad[1]), BigInt(paramsRender.pad[2]), BigInt(paramsRender.pad[3]), dirModel
	);

	const tensorAlpha3Padded = hasAlpha
		? await padding(tensorAlpha3, BigInt(paramsRender.pad[0]), BigInt(paramsRender.pad[1]), BigInt(paramsRender.pad[2]), BigInt(paramsRender.pad[3]))
		: { data: null };


	// const channelTile = tensorColorPadded.dims[1];
	const heightTile = tensorColorPadded.dims[2];
	const widthTile = tensorColorPadded.dims[3];

	// create temporary canvas for tile input
	const dataImageTileInput = toImageData(tensorColorPadded.data, tensorAlpha3Padded.data, widthTile, heightTile);
	const canvasTileInput = createCanvas(widthTile, heightTile);
	const contextTileInput = canvasTileInput.getContext('2d', { willReadFrequently: true });
	contextTileInput.putImageData(dataImageTileInput, 0, 0);


	const countBlocks = paramsRender.heightBlocks * paramsRender.widthBlocks;



	// tiled start render

	// create index list
	const tiles = [];
	for(let indexHeight = 0; indexHeight < paramsRender.heightBlocks; ++indexHeight) {
		for(let indexWidth = 0; indexWidth < paramsRender.widthBlocks; ++indexWidth) {
			const i = indexHeight * paramsRender.stepTileInput;
			const j = indexWidth * paramsRender.stepTileInput;
			const ii = indexHeight * paramsRender.stepTileOutput;
			const jj = indexWidth * paramsRender.stepTileOutput;

			tiles.push([i, j, ii, jj, indexHeight, indexWidth]);
		}
	}


	// shuffle tiled rendering
	if(isTileShuffle) { shuffleArray(tiles); }


	// setup output canvas
	const canvasOutput = createCanvas(canvasRaw.width * scale, canvasRaw.height * scale);
	const contextOutput = canvasOutput.getContext('2d', { willReadFrequently: true });



	let countBlockFinished = 0;
	emitBlockFinished(0, countBlocks, true);

	for(let k = 0; k < tiles.length; ++k) {
		const [i, j, ii, jj, iHeight, iWidth] = tiles[k];
		const dataImageTile = contextTileInput.getImageData(j, i, sizeTile, sizeTile);


		const colorSingle = checkSingleColor(dataImageTile.data, hasAlpha);


		/** @type {Onnx.Tensor} */
		let tensorTileColorTarget;
		/** @type {Onnx.Tensor | undefined} */
		let tensorTileAlphaTarget;
		if(colorSingle == null) {
			const [tensorTileColorFull, , tensorTileAlpha3Full] = convertImageDataToTensor(
				dataImageTile.data, dataImageTile.width, dataImageTile.height, hasAlpha
			);

			const tensorTileColor = tta > 0
				? await splitTTA(tensorTileColorFull, BigInt(tta), dirModel)
				: tensorTileColorFull;

			tensorTileColorTarget = tta > 0
				? await mergeTTA((await model.run({ x: tensorTileColor })).y, BigInt(tta), dirModel)
				: (await model.run({ x: tensorTileColor })).y;


			if(hasAlpha) { tensorTileAlphaTarget = (await modelAlpha.run({ x: tensorTileAlpha3Full })).y; }
		}
		// no need waifu2x, tile is single color image
		else {
			[tensorTileColorTarget, tensorTileAlphaTarget] = createSingleColorTensor(colorSingle, sizeTile * scale - offset * 2);
		}



		const colorUpdated = seamBlendingColor.update(tensorTileColorTarget, iHeight, iWidth);
		const alphaUpdated = hasAlpha ? seamBlendingAlpha.update(tensorTileAlphaTarget, iHeight, iWidth) : { data: null };

		const dataImageOutput = toImageData(colorUpdated.data, alphaUpdated.data, tensorTileColorTarget.dims[3], tensorTileColorTarget.dims[2]);

		contextOutput.putImageData(dataImageOutput, jj, ii);



		countBlockFinished++;

		emitBlockFinished(countBlockFinished, countBlocks, !controller.break);
		if(controller.break) { break; }
	}



	return new Promise((resolver, rejecter) =>
		canvasOutput.toBuffer((error, bufferFinal) =>
			error ? rejecter(error) : resolver(bufferFinal)
		)
	);
};
