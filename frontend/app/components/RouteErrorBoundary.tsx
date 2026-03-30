import { isRouteErrorResponse } from "@remix-run/react";
import { Card } from "~/components/Card";

export function RouteErrorBoundary({
  sport,
  error,
}: {
  sport: string;
  error: unknown;
}) {

  let message = `Something went wrong loading ${sport} data.`;
  let details = "Please try again later.";

  if (isRouteErrorResponse(error)) {
    message = `Error ${error.status}: Failed to load ${sport} data`;
    details =
      error.status === 500
        ? `The ${sport} backend is currently unavailable. Please try again in a few minutes.`
        : error.statusText || "Please try again later.";
  } else if (error instanceof Error) {
    details = error.message;
  }

  return (
    <div className="overflow-y-auto flex flex-col gap-4 p-6 pb-36 lg:pb-24 lg:px-12">
      <Card title={`${sport} Data Unavailable`}>
        <div className="flex flex-col gap-3">
          <p className="text-red-400 font-medium">{message}</p>
          <p className="text-gray-400">{details}</p>
          <button
            onClick={() => window.location.reload()}
            className="self-start px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded transition-colors"
          >
            Try Again
          </button>
        </div>
      </Card>
    </div>
  );
}
