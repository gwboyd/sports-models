export const displaySpread = (spread: number, numDecimals: number = 1) =>
  spread > 0 ? `+${spread.toFixed(numDecimals)}` : spread.toFixed(numDecimals);

export const convertDateTime = (dateTimeString: string): Date => {
  const [year, month, day, hour, minute] = dateTimeString
    .split(/[-:]/)
    .map(Number);
  return new Date(year, month - 1, day, hour, minute);
};
