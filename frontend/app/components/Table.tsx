"use client";
/* eslint-disable react-hooks/incompatible-library */

import type {
  ColumnDef,
  RowData,
  SortingState,
  TableMeta,
} from "@tanstack/react-table";
import {
  useReactTable,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
} from "@tanstack/react-table";
import { useState } from "react";

declare module "@tanstack/table-core" {
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  interface TableMeta<TData extends RowData> {
    bankroll?: number;
  }
}

export function Table<TData>({
  columns,
  data,
  stickyHeader = false,
  compact = false,
  meta,
}: {
  columns: ColumnDef<TData>[];
  data: TData[];
  stickyHeader?: boolean;
  compact?: boolean;
  meta?: TableMeta<TData>;
}) {
  const [sorting, setSorting] = useState<SortingState>([]);

  const table = useReactTable<TData>({
    data,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    meta,
  });

  return (
    <div
      className={`overflow-auto ${compact ? "" : "min-h-[350px]"} rounded-lg border ${theme.borderColor} bg-white`}
    >
      <table
        className={`table-auto w-full text-center 
            ${
              stickyHeader
                ? "border-separate border-spacing-0"
                : "border-collapse"
            }`}
      >
        <thead className={`${stickyHeader ? "sticky top-0 z-10" : ""}`}>
          {table.getHeaderGroups().map((headerGroup) => (
            <tr key={headerGroup.id}>
              {headerGroup.headers.map((header, index) => (
                <th
                  key={header.id}
                  className={`
                    ${theme.headerBackground} 
                    ${theme.headerText} 
                    ${theme.borderColor} 
                    border-b 
                    ${index === 0 ? "border-l" : ""} 
                    border-r 
                    p-2 
                    cursor-pointer 
                    select-none 
                    hover:bg-slate-100
                    transition-colors 
                    duration-[100ms]
                  `}
                  onClick={header.column.getToggleSortingHandler()}
                >
                  <div className="flex justify-center items-center gap-4">
                    {flexRender(
                      header.column.columnDef.header,
                      header.getContext()
                    )}
                    {header.column.getIsSorted() && (
                      <span className="text-xs">
                        {{
                          asc: "▲",
                          desc: "▼",
                        }[header.column.getIsSorted() as string] ?? null}
                      </span>
                    )}
                  </div>
                </th>
              ))}
            </tr>
          ))}
        </thead>
        <tbody>
          {table.getRowModel().rows.map((row) => (
            <tr key={row.id} className={`${theme.cellBackground} transition-colors hover:bg-slate-50`}>
              {row.getVisibleCells().map((cell, index) => (
                <td
                  key={cell.id}
                  className={`
                    ${theme.cellText} 
                    ${theme.borderColor} 
                    border-b 
                    ${index === 0 ? "border-l" : ""} 
                    border-r 
                    p-2
                  `}
                >
                  {flexRender(cell.column.columnDef.cell, cell.getContext())}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const theme = {
  borderColor: "border-slate-200",
  headerBackground: "bg-slate-50",
  headerText: "text-slate-700",
  cellBackground: "bg-white",
  cellText: "text-slate-700",
};
