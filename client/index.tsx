import React from "react";
import ReactDOM from "react-dom";
import FITEController from "./context";
import TextArea from "./components";
import useFite from "./hooks";

function MyComponents() {
  const { dispatchState, apiUrl, models, strategies, ...state } = useFite();

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
      {strategies.map((strategy) => (
        <div key={strategy}>
          <input
            type="radio"
            name="strategy"
            value={strategy}
            onChange={(e) => dispatchState({ ["strategy"]: e.target.value })}
          />
          <label>{strategy}</label>
        </div>
      ))}
    </div>
  );
}

ReactDOM.render(
  <React.StrictMode>
    <FITEController apiUrl="http://localhost:8000" model="gpt2-taf-base1" strategy="GREEDY">
      <MyComponents />
      <TextArea />
    </FITEController>
  </React.StrictMode>,
  document.getElementById("root")
);

// Hot Module Replacement (HMR) - Remove this snippet to remove HMR.
// Learn more: https://snowpack.dev/concepts/hot-module-replacement
if (import.meta.hot) {
  import.meta.hot.accept();
}
// export default FITE;
