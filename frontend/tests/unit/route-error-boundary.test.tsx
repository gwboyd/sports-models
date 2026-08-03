import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { RouteErrorBoundary } from "@/app/components/RouteErrorBoundary";

describe("RouteErrorBoundary", () => {
  it("retries through the Next.js error-boundary reset callback", () => {
    const reset = vi.fn();
    render(<RouteErrorBoundary sport="NFL" error={new Error("upstream failed")} reset={reset} />);

    fireEvent.click(screen.getByRole("button", { name: "Try Again" }));

    expect(reset).toHaveBeenCalledTimes(1);
    expect(screen.getByText("NFL Data Unavailable")).toBeVisible();
  });
});
