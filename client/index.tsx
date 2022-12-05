import React from "react";
import ReactDOM from "react-dom";
//
import Controller from "./context";
import { TextAreaStack, InputLayer, AutoCompletionLayer } from "./components";
import useFITE from "./hooks";

// a dropdown menu for selecting a model
function ModelSelector() {
  const { modelOptions, model, dispatchState } = useFITE();
  return (
    <select onChange={(e) => dispatchState({ model: e.target.value })} value={model}>
      {modelOptions &&
        modelOptions.map((model) => (
          <option key={model} value={model}>
            {model}
          </option>
        ))}
    </select>
  );
}
// a dropdown menu for selecting a strategy
function StrategySelector() {
  const { strategyOptions, strategy, dispatchState } = useFITE();
  return (
    <select onChange={(e) => dispatchState({ strategy: e.target.value })} value={strategy}>
      {strategyOptions &&
        strategyOptions.map((strategy) => (
          <option key={strategy} value={strategy}>
            {strategy}
          </option>
        ))}
    </select>
  );
}

function App() {
  return (
    <Controller apiUrl="http://localhost:8000" model="gpt2-taf-base1" strategy="GREEDY">
      {/*  the controller is a React.Context.provider*/}
      {/* the context provider propagates state and callbacks via the useFITE hook */}
      <ModelSelector />
      <StrategySelector />
      {/* the TextAreaStack is an aligned textarea and pre elements connected to state */}
      <TextAreaStack reRenderOnStrategyChange={true}>
        <AutoCompletionLayer />
        <InputLayer debounceWaitTime={600} />
      </TextAreaStack>
    </Controller>
  );
}

ReactDOM.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
  document.getElementById("root")
);

// Hot Module Replacement (HMR) - Remove this snippet to remove HMR.
// Learn more: https://snowpack.dev/concepts/hot-module-replacement
if (import.meta.hot) {
  import.meta.hot.accept();
}
// export default FITE;
