// @vitest-environment node
/// <reference types="node" />
import { readdirSync } from 'node:fs';
import { join, parse, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { expect, it } from 'vitest';

// Windows and macOS ignore letter case in file names, and imports omit the
// extension, so "./Lens" can resolve to lens.ts instead of Lens.tsx there
// (a blank page that Linux CI never sees). Keep module names unique ignoring
// case and extension.
const SRC = fileURLToPath(new URL('.', import.meta.url));

function modules(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) return modules(path);
    return /\.(ts|tsx|js|jsx)$/.test(entry.name) ? [path] : [];
  });
}

it('has no two modules whose import paths collide on case-insensitive file systems', () => {
  const seen = new Map<string, string>();
  const collisions: string[] = [];
  for (const file of modules(SRC)) {
    const { dir, name } = parse(relative(SRC, file));
    const key = join(dir, name).toLowerCase();
    if (seen.has(key)) collisions.push(`${seen.get(key)} <-> ${relative(SRC, file)}`);
    else seen.set(key, relative(SRC, file));
  }
  expect(collisions).toEqual([]);
});
