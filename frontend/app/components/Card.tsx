import type { ReactNode } from "react";

export const Card = ({
  title,
  className,
  children,
}: {
  title?: string;
  className?: string;
  children: ReactNode;
}) => (
  <div
    className={`rounded-lg border border-[var(--border)] bg-[var(--surface)] p-3.5 ${className ?? ""}`}
  >
    {title && <strong className="mb-2 block text-sm font-semibold text-[var(--ink)]">{title}</strong>}
    <div>{children}</div>
  </div>
);
