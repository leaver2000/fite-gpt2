import React from "react";
import ReactDOM from "react-dom";
//
import Controller from "./context";
import { TextAreaStack, TextInputLayer, TextCompletionLayer } from "./components";
import useFITE from "./hooks";

const Selections = {
  model: { value: "model", options: "modelOptions" },
  strategy: { value: "strategy", options: "strategyOptions" },
} as const;

interface SelectorProps {
  selection: typeof Selections.model | typeof Selections.strategy;
}

function Selector({ selection }: SelectorProps): JSX.Element {
  const fite = useFITE();
  const [value, options] = [fite[selection.value], fite[selection.options]];

  return (
    <li style={{ display: "inline" }}>
      <select onChange={(e) => fite.setPartialState({ [selection.value]: e.target.value })} value={value}>
        {options &&
          options.map((value: string) => (
            <option key={value} value={value}>
              {value}
            </option>
          ))}
      </select>
    </li>
  );
}

function Navbar() {
  const style = {
    listStyle: "none",
    margin: 0,
    padding: 2.5,
    backgroundColor: "#333",
  };
  return (
    <ul style={style}>
      <Selector selection={Selections.model} />
      <Selector selection={Selections.strategy} />
    </ul>
  );
}

function App() {
  return (
    <div style={{ height: "100vh", width: "100vw" }}>
      {/*
       the Controller is a Context.Provider that propagates via the useFITE hook 
       the props are passed to the Controller and then to the Context.Provider
       if no props are passed, the Controller will use the default props
       */}
      <Controller apiUrl="http://localhost:8000" model="gpt2-taf-base1" strategy="GREEDY">
        {/* a simple navbar to cycle the stateful api options */}
        <Navbar />
        {/* TextAreaStack vertically aligns the children in the stack */}
        <TextAreaStack reRenderOnStrategyChange={true}>
          {/* TextCompletionLayer sits behind the TextInputLayer to display the generated text */}
          <TextCompletionLayer />
          {/* TextInputLayer wraps a textarea with specialized handlers */}
          <TextInputLayer debounceWaitTime={600} toUpperCase={true} />
        </TextAreaStack>
      </Controller>
    </div>
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
