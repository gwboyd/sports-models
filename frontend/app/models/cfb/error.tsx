"use client";

import { RouteErrorBoundary } from "@/app/components/RouteErrorBoundary";

export default function CfbError({ error, reset }: Readonly<{ error: Error; reset: () => void }>) {
  return <RouteErrorBoundary sport="CFB" error={error} reset={reset} />;
}
