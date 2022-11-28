/** @type {import("snowpack").SnowpackUserConfig } */
export default {
    mount: {
      client :  { url: '/dist' },
      public: { url: '/', static: true },
    },
    plugins: [
      '@snowpack/plugin-react-refresh',
      '@snowpack/plugin-dotenv',
      [
        '@snowpack/plugin-typescript',
        {
          /* Yarn PnP workaround: see https://www.npmjs.com/package/@snowpack/plugin-typescript */
          ...(process.versions.pnp ? { tsc: 'yarn pnpify tsc' } : {}),
        },
      ],
    ],
    exclude: [
      '**/node_modules/**/*',
      "src/server/**/*",
      "src/model/**/*",
    ],
    env: {
      API_URL: 'api.google.com',
    },
    routes: [
      /* Enable an SPA Fallback in development: */
      // {src:""}
      // { "match": "routes", "client/src/": ".*", "dest": "/index.html" },
    ],
    optimize: {
      /* Example: Bundle your final build: */
      // "bundle": true,
    },
    packageOptions: {
      /* ... */
    },
    devOptions: {
      /* ... */
    },
    buildOptions: {
      /* ... */
    },
  };
  