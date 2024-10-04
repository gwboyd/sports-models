export const displaySpread = (spread: number, numDecimals: number = 1) =>
  spread > 0 ? `+${spread.toFixed(numDecimals)}` : spread.toFixed(numDecimals);
