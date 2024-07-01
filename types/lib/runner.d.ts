export function calcOffset(arch: ArchOption, scale: ScaleOption): number;
export function calcTileSize(arch: ArchOption, tile: TileOption, offset?: number | undefined): number;
export function getUtilOnnxSession(name: string, dirModel: string): Promise<Onnx.InferenceSession>;
export function convertImageDataToTensor(rgba: Uint8ClampedArray, width: number, height: number, keepAlpha?: boolean): [Onnx.TypedTensor<"float32">, Onnx.TypedTensor<"float32">, Onnx.TypedTensor<"float32">];
export const SeamBlending: {
    new (sizesSource: number[], scale: number, offset: number, tile: number, sizeBlend: number | undefined, dirModel: string): {
        /** @type {number[]} */
        sizesSource: number[];
        /** @type {number} */
        scale: number;
        /** @type {number} */
        offset: number;
        /** @type {number} */
        tile: number;
        /** @type {number} */
        sizeBlend: number;
        /** @type {Onnx.Tensor} */
        filterBlend: Onnx.Tensor;
        /** @type {SeamBlendingParam} */
        param: SeamBlendingParam;
        /** @type {Onnx.TypedTensor<"float32">} */
        pixels: Onnx.TypedTensor<"float32">;
        /** @type {Onnx.TypedTensor<"float32">} */
        weights: Onnx.TypedTensor<"float32">;
        /** @type {Onnx.TypedTensor<"float32">} */
        output: Onnx.TypedTensor<"float32">;
        calcParams(): void;
        /**
         * @param {Onnx.Tensor} tensorSource
         * @param {number} iTile
         * @param {number} jTile
         * @returns {Onnx.TypedTensor<"float32">}
         */
        update(tensorSource: Onnx.Tensor, iTile: number, jTile: number): Onnx.TypedTensor<"float32">;
        /**
         * @param {string} dirModel
         * @returns {Onnx.Tensor}
         */
        createSeamBlendingFilter(dirModel: string): Onnx.Tensor;
    };
    BLEND_SIZE: number;
};
export function padAlphaBorder(tensorColor: Onnx.Tensor, tensorAlpha: Onnx.Tensor, offset: bigint, dirModel: string): Promise<Onnx.TypedTensor<"float32">>;
export function padding(tensorSource: Onnx.Tensor, left: BigInt, right: BigInt, top: BigInt, bottom: BigInt, dirModel: string): Promise<Onnx.Tensor>;
export function toImageData(z: Float32Array, alpha3: Float32Array, width: number, height: number): ImageData;
export function shuffleArray(array: any[]): void;
export function checkSingleColor(rgba: Uint8ClampedArray, keepAlpha?: boolean | undefined): [number, number, number, number];
export function splitTTA(tensorSource: Onnx.Tensor, tta: number, dirModel: string): Promise<Onnx.Tensor>;
export function mergeTTA(tensorSource: Onnx.Tensor, tta: number, dirModel: string): Promise<Onnx.Tensor>;
export function createSingleColorTensor(rgba: number[], size: number): [Onnx.TypedTensor<"float32">, Onnx.TypedTensor<"float32">];
export function checkAlphaChannel(rgba: Uint8ClampedArray): boolean;
export type ArchOption = import("../bases.d.ts").ArchOption;
export type ScaleOption = import("../bases.d.ts").ScaleOption;
export type TileOption = import("../bases.d.ts").TileOption;
export type SeamBlendingParam = import("../bases.d.ts").SeamBlendingParam;
import Onnx from 'onnxruntime-node';
import { ImageData } from 'canvas';
