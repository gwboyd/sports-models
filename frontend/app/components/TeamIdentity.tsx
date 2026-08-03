"use client";

import { useState } from "react";
import type { TeamIdentity as TeamIdentityValue } from "@/app/lib/team-data";

export function TeamIdentity({
  team,
  compact = false,
  showName = true,
}: {
  team: TeamIdentityValue;
  compact?: boolean;
  showName?: boolean;
}) {
  const [failedPath, setFailedPath] = useState<string | undefined>();
  const failed = failedPath === team.logoPath;
  const size = compact ? "h-8 w-8 text-[10px]" : "h-10 w-10 text-xs";

  return (
    <span className="inline-flex min-w-0 items-center gap-2.5">
      <span className={`flex shrink-0 items-center justify-center overflow-hidden rounded-full border border-slate-200 bg-slate-100 font-bold text-slate-600 ${size}`}>
        {team.logoPath && !failed ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={team.logoPath} alt="" className="h-[82%] w-[82%] object-contain" onError={() => setFailedPath(team.logoPath)} />
        ) : team.abbreviation.slice(0, 4)}
      </span>
      {showName ? <span className="truncate font-medium text-[var(--ink)]">{team.displayName}</span> : null}
    </span>
  );
}
