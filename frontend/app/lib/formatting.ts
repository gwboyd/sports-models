export const displaySpread = (spread: number, numDecimals = 1) =>
  spread > 0 ? `+${spread.toFixed(numDecimals)}` : spread.toFixed(numDecimals);

export const displayProbability = (probability: number) => `${probability.toFixed(1)}%`;

const FOOTBALL_SOURCE_TIME_ZONE = "America/New_York";

function dateTimeParts(dateTimeString: string) {
  const [year, month, day, hour, minute] = dateTimeString.split(/[-:]/).map(Number);
  return { year, month, day, hour, minute };
}

function validDateTimeParts(value: ReturnType<typeof dateTimeParts>): boolean {
  return Object.values(value).every(Number.isFinite);
}

function timeZoneOffset(date: Date, timeZone: string): number {
  const values = Object.fromEntries(new Intl.DateTimeFormat("en-US", {
    timeZone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hourCycle: "h23",
  }).formatToParts(date).map((part) => [part.type, part.value]));
  return Date.UTC(
    Number(values.year),
    Number(values.month) - 1,
    Number(values.day),
    Number(values.hour),
    Number(values.minute),
    Number(values.second),
  ) - date.getTime();
}

function newYorkWallTimeToDate(dateTimeString: string): Date {
  const value = dateTimeParts(dateTimeString);
  if (!validDateTimeParts(value)) return new Date(Number.NaN);
  const wallTime = Date.UTC(value.year, value.month - 1, value.day, value.hour, value.minute);
  let instant = wallTime;
  for (let attempt = 0; attempt < 2; attempt += 1) {
    instant = wallTime - timeZoneOffset(new Date(instant), FOOTBALL_SOURCE_TIME_ZONE);
  }
  return new Date(instant);
}

export function convertDateTime(dateTimeString: string): Date {
  return newYorkWallTimeToDate(dateTimeString);
}

export function formatKickoff(dateTimeString: string, timeZone?: string): string {
  const parsed = convertDateTime(dateTimeString);
  if (Number.isNaN(parsed.getTime())) return dateTimeString;
  return new Intl.DateTimeFormat("en-US", {
    timeZone: timeZone ?? FOOTBALL_SOURCE_TIME_ZONE,
    weekday: "short",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(parsed);
}

export function formatGameDate(dateTimeString: string, timeZone?: string): string {
  const parsed = convertDateTime(dateTimeString);
  if (Number.isNaN(parsed.getTime())) return dateTimeString;
  return new Intl.DateTimeFormat("en-US", {
    timeZone: timeZone ?? FOOTBALL_SOURCE_TIME_ZONE,
    weekday: "long",
    month: "long",
    day: "numeric",
  }).format(parsed);
}

export function formatUpdatedAt(writeTime: string, timeZone = "UTC"): string {
  const hasTimezone = writeTime.endsWith("Z") || /[+-]\d\d:\d\d$/.test(writeTime);
  const parsed = new Date(writeTime.replace(" ", "T") + (hasTimezone ? "" : "Z"));
  if (Number.isNaN(parsed.getTime())) return writeTime;
  return new Intl.DateTimeFormat("en-US", {
    timeZone,
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    timeZoneName: "short",
  }).format(parsed);
}
