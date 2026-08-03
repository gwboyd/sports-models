export const displaySpread = (spread: number, numDecimals = 1) =>
  spread > 0 ? `+${spread.toFixed(numDecimals)}` : spread.toFixed(numDecimals);

export const displayProbability = (probability: number) => `${probability.toFixed(1)}%`;

const SHORT_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const LONG_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
const SHORT_WEEKDAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const LONG_WEEKDAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];

function dateTimeParts(dateTimeString: string) {
  const [year, month, day, hour, minute] = dateTimeString.split(/[-:]/).map(Number);
  return { year, month, day, hour, minute, weekday: new Date(Date.UTC(year, month - 1, day)).getUTCDay() };
}

function displayClock(hour: number, minute: number): string {
  const suffix = hour >= 12 ? "PM" : "AM";
  const twelveHour = hour % 12 || 12;
  return `${twelveHour}:${String(minute).padStart(2, "0")} ${suffix}`;
}

export function formatKickoff(dateTimeString: string): string {
  const value = dateTimeParts(dateTimeString);
  return `${SHORT_WEEKDAYS[value.weekday]}, ${SHORT_MONTHS[value.month - 1]} ${value.day}, ${displayClock(value.hour, value.minute)}`;
}

export function formatGameDate(dateTimeString: string): string {
  const value = dateTimeParts(dateTimeString);
  return `${LONG_WEEKDAYS[value.weekday]}, ${LONG_MONTHS[value.month - 1]} ${value.day}`;
}

export function formatUpdatedAt(writeTime: string): string {
  const hasTimezone = writeTime.endsWith("Z") || /[+-]\d\d:\d\d$/.test(writeTime);
  const parsed = new Date(writeTime.replace(" ", "T") + (hasTimezone ? "" : "Z"));
  if (Number.isNaN(parsed.getTime())) return writeTime;
  return `${SHORT_MONTHS[parsed.getUTCMonth()]} ${parsed.getUTCDate()}, ${displayClock(parsed.getUTCHours(), parsed.getUTCMinutes())} UTC`;
}

export const convertDateTime = (dateTimeString: string): Date => {
  const [year, month, day, hour, minute] = dateTimeString.split(/[-:]/).map(Number);
  return new Date(year, month - 1, day, hour, minute);
};
