import { fromArrayBuffer } from "geotiff";

/**
 * Parse a GeoTIFF byte buffer into width, height and Float32Array pixel data.
 * @param {ArrayBuffer} buffer
 * @returns {Promise<{ width: number, height: number, data: Float32Array }>}
 */
export async function parseTiff(buffer) {
  const tiff = await fromArrayBuffer(buffer);
  const image = await tiff.getImage();
  const width = image.getWidth();
  const height = image.getHeight();
  const [raster] = await image.readRasters();
  const data = new Float32Array(raster);
  return { width, height, data };
}
