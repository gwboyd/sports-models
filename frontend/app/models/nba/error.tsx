"use client";

import { RouteErrorBoundary } from "@/app/components/RouteErrorBoundary";

export default function NbaError({ error, reset }: Readonly<{ error: Error; reset: () => void }>) {
  return <RouteErrorBoundary sport="NBA" error={error} reset={reset} />;
}
