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
    <div className="overflow-y-auto flex flex-col gap-4 p-6 pb-36 lg:pb-24 lg:px-12">
      <Card title={`${sport} Data Unavailable`}>
        <div className="flex flex-col gap-3">
          <p className="text-red-400 font-medium">{message}</p>
          <p className="text-gray-400">{details}</p>
          <button
            onClick={() => {
              if (reset) {
                reset();
              } else {
                window.location.reload();
              }
            }}
            className="self-start px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded transition-colors"
          >
            Try Again
          </button>
        </div>
      </Card>
    </div>
  );
}
