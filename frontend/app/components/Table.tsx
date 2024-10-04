import type { ColumnDef, SortingState } from "@tanstack/react-table";
import {
  useReactTable,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
} from "@tanstack/react-table";
import { useState } from "react";

export function Table<TData>({
  columns,
  data,
  stickyHeader = false,
}: {
  columns: ColumnDef<TData, any>[];
  data: TData[];
  stickyHeader?: boolean;
}) {
  const [sorting, setSorting] = useState<SortingState>([]);

  const table = useReactTable<TData>({
    data,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  return (
    <div
      className={`overflow-x-auto min-h-[350px] border-y ${theme.borderColor}`}
    >
      <table className="table-auto border-collapse w-full text-center">
        <thead className={`${stickyHeader ? "sticky top-0 z-10" : ""}`}>
          {table.getHeaderGroups().map((headerGroup) => (
            <tr key={headerGroup.id}>
              {headerGroup.headers.map((header) => (
                <th
                  key={header.id}
                  className={`border-x ${theme.borderColor} p-2 ${theme.headerBackground} ${theme.headerText} cursor-pointer select-none hover:bg-gray-500 transition-colors duration-[100ms]`}
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
            <tr key={row.id} className={theme.cellBackground}>
              {row.getVisibleCells().map((cell) => (
                <td
                  key={cell.id}
                  className={`border ${theme.borderColor} p-2 ${theme.cellText}`}
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
  borderColor: "border-gray-600",
  headerBackground: "bg-gray-700",
  headerText: "text-white",
  cellBackground: "bg-gray-800",
  cellText: "text-gray-300",
};
