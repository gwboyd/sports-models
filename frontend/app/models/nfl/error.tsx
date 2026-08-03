"use client";

import { RouteErrorBoundary } from "@/app/components/RouteErrorBoundary";

export default function NflError({ error, reset }: Readonly<{ error: Error; reset: () => void }>) {
  return <RouteErrorBoundary sport="NFL" error={error} reset={reset} />;
}
