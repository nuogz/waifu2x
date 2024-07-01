export type ArchOption = "swin_unet" | "cunet";
export type StyleOption = "art" | "art_scan" | "photo";
export type NoiseOption = -1 | 0 | 1 | 2 | 3;
export type ScaleOption = 1 | 2 | 4;
export type TileOption = 64 | 256;
export type TTAOption = 0 | 2 | 4;

export type Waifu2xOption = {
	arch: ArchOption;
	style: StyleOption;
	noise: NoiseOption;
	scale: ScaleOption;
	tile: TileOption;
	isTileShuffle: boolean;
	tta: TTAOption;
	enableAlphaChannel: boolean;
	dirModel: string;
};
export type Controller = {
	break: boolean;
};
export type BlockFinishedHandle = (countFinishedNow: number, countFinishedMax: number, stillProcessing: boolean) => any;


export type SeamBlendingParam = {
	heightTarget: number;
	widthTarget: number;
	offsetInput: number;
	sizeBlendInput: number;
	stepTileInput: number;
	stepTileOutput: number;
	heightBlocks: number;
	widthBlocks: number;
	heightBufferTarget: number;
	widthBufferTarget: number;
	pad: number[];
};
