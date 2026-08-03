"use client";

import { Table } from "@/app/components/Table";
import type { NBAFirstBasketPick } from "@/app/types/types";
import { columns } from "./Cols";

export function FirstBasketTable({ data, bankroll }: { data: NBAFirstBasketPick[]; bankroll: number }) {
  return <Table stickyHeader columns={columns} data={data} meta={{ bankroll }} />;
}
