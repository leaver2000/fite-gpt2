import React from "react";
import debounce from "lodash.debounce";
import useFite from "./hooks";
import "./style.css";

enum ActionKey {
  TAB = "Tab",
  ENTER = "Enter",
  // UP = "ArrowUp",
  // DOWN = "ArrowDown",
  // BACKSPACE = "Backspace",
}
const ActionKeys = Object.values(ActionKey) as string[];

// console.log(MagicKeys["ArrowUp"])
// console.log(MagicKeys)
// console.log(MagicKeys.UP)
function FITE() {
  /* a text area with intellisense autocomplete
    that is used for writing a terminal aerodrome forecast
    the intellisense is based the TAF_STRING above
    the user can cycle through the intellisense options with the up and down arrow keys
    the user can select an option by pressing tab or enter */
  const { dispatchState, generateText, ...state } = useFite();
  const { model, strategy, textPrompt, textCompletion, textAreaValue } = state;

  const handleKeyDown = React.useCallback<(e: React.KeyboardEvent<HTMLTextAreaElement>) => void>(
    // event handler for the text area manages the intellisense options and the text area value
    (e) => {
      // continue on any non special key; cannot perform any action if there is no text completion
      if (!textCompletion) return void 0;
      // unpack the HTMLTextAreaElementEvent
      const { key, ctrlKey, currentTarget } = e;
      // and currentTarget.value as targetValue
      let { value: targetValue } = currentTarget;
      // ActionKey logic...
      if (ActionKeys.includes(key) && textAreaValue) {
        // [ TAB ] -> update the targetValue with the first word of the textCompletion
        if (key === ActionKey.TAB) {
          e.preventDefault();
          // substring the textCompletion to the first word not in the targetValue and split it into an array
          const textList = textCompletion.substring(targetValue.length, textCompletion.length).split(" ");
          // dispatch the new textAreaValue to the state as state.textAreaValue
          dispatchState({ textAreaValue: `${targetValue.trim()} ${textList.find((s) => !!s)}` });
          // [ ENTER && CTRL + ENTER] -> ...
        } else if (key === ActionKey.ENTER) {
          e.preventDefault();
          // [ CTRL + ENTER ] -> update the targetValue with the full textCompletion
          if (ctrlKey) {
            targetValue = textCompletion;
            // [ ENTER ] -> update the targetValue with the first sentence of the textCompletion
          } else {
            targetValue += textCompletion.substring(targetValue.length, textCompletion.length).split("\n")[0] + "\n";
          }
          dispatchState({ textAreaValue: targetValue });
        } else {
          // raise an error if the key is not one of the above
          throw new Error(`key ${key} is not a valid ActionKey; refer to ActionKeys`);
        }
      }
    },
    [textPrompt, textCompletion, textAreaValue]
  );
  const bounce = React.useCallback(
    debounce(
      (text: string) => {
        // const { value } = e.target;
        generateText(text, model, strategy).then((completion) =>
          dispatchState({ textCompletion: completion.join("\n") })
        );
      },
      1000,
      { leading: false, trailing: true }
    ),
    [model, strategy]
  );
  const handleOnTextAreaChange = React.useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      const { value } = e.target;
      bounce(value);
      console.log([e.target.value, e.target.selectionStart, e.target.selectionEnd]);
      dispatchState({ textAreaValue: value });
    },
    [bounce]
  );

  return (
    <div className="taf-component-container">
      <div style={{ position: "absolute" }}>
        <pre className="taf-annotation-layer">{textCompletion}</pre>
      </div>
      <textarea
        className="taf-input-layer"
        onKeyDown={handleKeyDown}
        onChange={handleOnTextAreaChange}
        value={textAreaValue}
      />
    </div>
  );
}

export default FITE;
