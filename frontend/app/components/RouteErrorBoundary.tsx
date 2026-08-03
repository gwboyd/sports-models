"use client";

import { Card } from "@/app/components/Card";

export function RouteErrorBoundary({
  sport,
  error,
  reset,
}: {
  sport: string;
  error?: Error;
  reset?: () => void;
}) {
  const message = `Something went wrong loading ${sport} data.`;
  let details = "Please try again later.";

  if (error?.message && process.env.NODE_ENV !== "production") {
    details = error.message;
  }

  return (
    <div className="mx-auto flex w-full max-w-4xl flex-col gap-4 px-4 py-12 sm:px-6 lg:px-8">
      <Card title={`${sport} Data Unavailable`}>
        <div className="flex flex-col gap-3">
          <p className="font-medium text-[var(--danger)]">{message}</p>
          <p className="text-[var(--muted)]">{details}</p>
          <button
            onClick={() => {
              if (reset) {
                reset();
              } else {
                window.location.reload();
              }
            }}
            className="min-h-11 self-start rounded-xl bg-[var(--accent)] px-4 py-2 font-semibold text-white transition-colors hover:bg-blue-700"
          >
            Try Again
          </button>
        </div>
      </Card>
    </div>
  );
}
