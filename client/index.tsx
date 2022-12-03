import React from "react";
import ReactDOM from "react-dom";
//
import Controller from "./context";
import { TextAreaStack, InputLayer, AnnotationLayer } from "./components";
import useFite from "./hooks";

function MyComponents() {
  //
  const { models, strategies, dispatchState } = useFite();

  return (
    <div>
      {/* for each model add a radio button */}
      {models &&
        models.map((model) => (
          <div key={model}>
            <input type="radio" name="model" value={model} onChange={(e) => dispatchState({ model: e.target.value })} />
            <label>{model}</label>
          </div>
        ))}
      {/* for each strategy add a radio button */}
      {strategies &&
        strategies.map((strategy) => (
          <div key={strategy}>
            <input
              type="radio"
              name="strategy"
              value={strategy}
              onChange={(e) => dispatchState({ strategy: e.target.value })}
            />
            <label>{strategy}</label>
          </div>
        ))}
    </div>
  );
}

function App() {
  return (
    /** the controller is a React Context.provider */
    <Controller apiUrl="http://localhost:8000" model="gpt2-taf-base1" strategy="GREEDY">
      {/* the context provider propagates state and callbacks via the useFite hook */}
      <MyComponents />
      {/* the TextAreaStack is an aligned textarea and pre elements connected to state */}
      <TextAreaStack>
        <AnnotationLayer />
        <InputLayer />
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
