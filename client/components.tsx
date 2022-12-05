import React from "react";
import debounce from "lodash.debounce";
import useFITE from "./hooks";
import "./style.css";

const ACTIONS = {
  TAB: "Tab",
  ENTER: "Enter",
  contain: function (key: string) {
    return Object.values(this)
      .filter((v) => typeof v === "string")
      .includes(key);
  } as (key: string) => boolean,
} as const;

enum ClassNames {
  inputLayer = "fite-input-layer",
  annotationLayer = "fite-annotation-layer",
  textAreaStack = "fite-text-area-stack",
}

function prepareTextAreaValue(textAreaValue: string): string {
  // split each line into an array -> reduce the text -> join the text
  return textAreaValue
    .split("\n")
    .reduce((previousValue: string[], currentValue: string) => {
      // remove preceding spaces or more than one trailing space
      currentValue = currentValue.replace(/^\s+|(\s)\s+$/, "");
      // remove empty lines
      return currentValue ? [...previousValue, currentValue] : previousValue;
    }, [])
    .join("\n");
}

type InputLayerProps = { debounceWaitTime?: number; toUpperCase?: boolean };

/**
 * the InputLayer component wraps a HTMLTextAreaElement
 */
export function InputLayer({ debounceWaitTime = 500, toUpperCase = true }: InputLayerProps): JSX.Element {
  const fite = useFITE();

  const handelTextAreaKeyDown = React.useCallback<(e: React.KeyboardEvent<HTMLTextAreaElement>) => void>(
    // event handler for the text area manages the intellisense options and the text area value
    (event) => {
      // continue if key not in ActionKeys or textCompletion is undefined
      if (!ACTIONS.contain(event.key)) return void 0;
      // prevent default action for ACTION keys
      event.preventDefault();
      // unpack the HTMLTextAreaElementEvent
      const { key, ctrlKey, currentTarget: { value: textPrompt } } = event; // prettier-ignore
      // slice the completion by the length of the prompt to get just the new generated text
      let textCompletion = fite.textCompletion.slice(textPrompt.length);
      // logic ...
      // [ TAB ] -> update the targetValue with the first word of the textCompletion
      if (key === ACTIONS.TAB) {
        // the index of the first space in the textCompletion that is not the first character
        const indexOfEndOfFirstWord = textCompletion.indexOf(" ", textCompletion.startsWith(" ") ? 1 : 0);
        // textAreaValue += the first whole word in the textCompletion
        textCompletion = textCompletion.substring(0, indexOfEndOfFirstWord);
      } // [ CTRL + ENTER ] -> full textCompletion
      else if (key === ACTIONS.ENTER && ctrlKey) void 0;
      // [ ENTER ] -> first full line of textCompletion
      else if (key === ACTIONS.ENTER && !ctrlKey) {
        textCompletion = textCompletion.substring(0, textCompletion.length).split("\n")[0] + "\n";
      } else throw new Error(`key ${key} is not a valid ${ACTIONS}; refer to ACTIONS`);
      fite.dispatchState({ textAreaValue: textPrompt + textCompletion });
    },
    [fite.textCompletion]
  );
  // debounce the generateText function to prevent spamming the server
  const debouncedTextCompletionDispatch = React.useCallback(
    debounce(
      async (textPrompt: string) => {
        // setting initial pending and count state
        let pending = true;
        let count = 0;
        // call to the api to generate text the response is a Promise<string[]>
        const results = fite.generateText(textPrompt).then((textArray) => {
          // update pending state now that the Promise has resolved
          pending = false;
          // join the textArray into a string separated by new lines
          return textArray.join("\n");
        });
        // while the results are pending; add '...' to the end of the textPrompt and dispatch the textCompletion
        while (pending) {
          // if the count is greater break the loop
          if (count > 10) {
            // log a warning
            console.warn("textCompletion took too long to resolve");
            break;
          }
          // update the textCompletion with the current textPrompt + the number of dots with a max of 3
          fite.dispatchState({ textCompletion: textPrompt.trim() + ".".repeat(count % 4) });
          // sleep for 250ms before dispatching the next textCompletion
          await new Promise((resolve) => setTimeout(resolve, 250));
          // increment count
          count++;
        }
        // dispatch the textCompletion with the results
        fite.dispatchState({ textCompletion: await results });
      },
      debounceWaitTime,
      { leading: false, trailing: true }
    ),
    [debounceWaitTime]
  );

  const handleOnTextAreaChange = React.useCallback(
    ({ target: { value } }: React.ChangeEvent<HTMLTextAreaElement>) => {
      const textAreaValue = prepareTextAreaValue(toUpperCase ? value.toUpperCase() : value);
      // if the textCompletion is empty or if the start of textCompletion is different from the textAreaValue
      if (!fite.textCompletion || !fite.textCompletion.startsWith(textAreaValue)) {
        fite.dispatchState({ textCompletion: textAreaValue });
        // make a call to the api using the lodash.debounce callback to dispatch the textCompletion
        debouncedTextCompletionDispatch(textAreaValue);
      }
      // always dispatch the textAreaValue
      fite.dispatchState({ textAreaValue });
    },
    [fite.textCompletion, fite.textAreaValue, debouncedTextCompletionDispatch, toUpperCase]
  );

  return (
    <textarea
      className={ClassNames.inputLayer}
      onKeyDown={handelTextAreaKeyDown}
      onChange={handleOnTextAreaChange}
      value={fite.textAreaValue}
    />
  );
}

export function AnnotationLayer({ children }: { children: React.ReactNode }): JSX.Element {
  return (
    <div style={{ position: "absolute" }}>
      <pre className={ClassNames.annotationLayer}>{children}</pre>
    </div>
  );
}

export function AutoCompletionLayer(): JSX.Element {
  const fite = useFITE();
  return <AnnotationLayer>{fite.textCompletion}</AnnotationLayer>;
}

type TextAreaStackProps = { reRenderOnStrategyChange?: boolean; children?: JSX.Element | JSX.Element[] };
export function TextAreaStack({ children, reRenderOnStrategyChange = true }: TextAreaStackProps): JSX.Element {
  const fite = useFITE();

  // if the strategy changes, re-render the textAreaValue
  React.useEffect(() => {
    if (reRenderOnStrategyChange) renderTextCompletion();
  }, [fite.strategy, reRenderOnStrategyChange]);

  // the callback is needed to prevent constant re-rendering on change to the textAreaValue
  const renderTextCompletion = React.useCallback<() => void>(() => {
    if (!fite.textAreaValue) return void 0;
    fite
      .generateText(fite.textAreaValue)
      .then((completion) => fite.dispatchState({ textCompletion: completion.join("\n") }));
  }, [fite.textAreaValue]);

  return <div className={ClassNames.textAreaStack}>{children}</div>;
}

export default function () {
  /* a text area with intellisense autocomplete
    that is used for writing a terminal aerodrome forecast
    the intellisense is based the TAF_STRING above
    the user can cycle through the intellisense options with the up and down arrow keys
    the user can select an option by pressing tab or enter */

  return (
    <TextAreaStack>
      <AutoCompletionLayer />
      <InputLayer />
    </TextAreaStack>
  );
}
