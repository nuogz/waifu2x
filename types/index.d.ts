/**
 * @param {Buffer} buffer
 * @param {Waifu2xOption} option
 * @param {BlockFinishedHandle} emitBlockFinished
 * @param {Controller} controller
 * @returns {Promise<Buffer>}
 */
export default function waifu2x(buffer: Buffer, option: Waifu2xOption, emitBlockFinished?: BlockFinishedHandle, controller?: Controller): Promise<Buffer>;
export type Waifu2xOption = import("./bases.d.ts").Waifu2xOption;
export type ArchOption = import("./bases.d.ts").ArchOption;
export type ScaleOption = import("./bases.d.ts").ScaleOption;
export type Controller = import("./bases.d.ts").Controller;
export type BlockFinishedHandle = import("./bases.d.ts").BlockFinishedHandle;
