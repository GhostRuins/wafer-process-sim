/// <reference types="vite/client" />

declare module 'd3-scale-chromatic' {
  export function interpolateViridis(t: number): string
  export function interpolateRdBu(t: number): string
  export function interpolateRdYlGn(t: number): string
}

interface ImportMetaEnv {
  readonly VITE_API_URL?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
