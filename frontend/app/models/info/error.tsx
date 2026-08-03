"use client";

import { RouteErrorBoundary } from "@/app/components/RouteErrorBoundary";

export default function InfoError({ error, reset }: Readonly<{ error: Error; reset: () => void }>) {
  return <RouteErrorBoundary sport="Info" error={error} reset={reset} />;
}
