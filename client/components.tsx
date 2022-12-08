import React from "react";
import debounce from "lodash.debounce";
import useFITE from "./hooks";
import { TAF_HEADER, TAF_HEADER_LENGTH } from "./context";
import "./style.css";
/**
 * 
 */
const ACTIONS = {
  TAB: "Tab",
  ENTER: "Enter",
  contain: function (key: string) {
    return Object.values(this)
      .filter((v) => typeof v === "string")
      .includes(key);
  } as (key: string) => boolean,
} as const;
/**
 * 
 */
enum ClassNames {
  inputLayer = "fite-input-layer",
  annotationLayer = "fite-annotation-layer",
  textAreaStack = "fite-text-area-stack",
}
/**
 * 
 * @param {string} textAreaValue 
 * @returns {string}
 */
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

type InputLayerProps = { debounceWaitTime?: number; toUpperCase?: boolean; textValueHeader?: string };
/**
 * The TextInputLayer component wraps a HTMLTextAreaElement
 * with specialized handlers for onChange and onKeyDown events
 * @param {number} debounceWaitTime - the time in milliseconds to wait before calling the onChange handler
 * @param {boolean} toUpperCase - if true, the text will be converted to uppercase
 */
export function TextInputLayer({
  debounceWaitTime = 500,
  toUpperCase = true,
}: InputLayerProps): JSX.Element {
  const fite = useFITE();
  // the textAreaRef is used to set the cursor position
  const textAreaRef = React.useRef<HTMLTextAreaElement>(null);
  const handelTextAreaKeyDown = React.useCallback<(e: React.KeyboardEvent<HTMLTextAreaElement>) => void>(
    // event handler for the text area manages the intellisense options and the text area value
    (event) => {
      // unpack the HTMLTextAreaElementEvent
      const { key, ctrlKey, currentTarget: { value: textPrompt, selectionStart } } = event; // prettier-ignore
      // prevent the user from deleting the header
      if (selectionStart < TAF_HEADER_LENGTH) {
        event.preventDefault();
        // set the cursor position to the end of the textPrompt
        textAreaRef.current!.selectionStart = textPrompt.length + 1;
        textAreaRef.current!.selectionEnd = textPrompt.length + 1;
        // set the textAreaValue to the textPrompt + a space
        fite.setPartialState({ textAreaValue: textPrompt.trim() + " " });
        return;
      } // if the key is not in the ACTIONS object, continue with the event
      else if (!ACTIONS.contain(event.key)) return;
      // prevent the default action for the key
      event.preventDefault();
      // slice the completion by the length of the prompt to get just the new generated text
      let textCompletion = fite.textCompletion.slice(textPrompt.length);
      /* 
      ACTION LOGIC
      these are the keyboard bindings for the textArea, most involve 
      updating the textAreaValue with some portion of the textCompletion
      - [ TAB ] -> update the targetValue with the first word of the textCompletion
      - [ CTRL + ENTER ] -> full textCompletion
      - [ ENTER ] -> first full line of textCompletion
      */
      // [ TAB ] -> update the targetValue with the first word of the textCompletion
      if (key === ACTIONS.TAB) {
        // the index of the first space in the textCompletion that is not the first character
        const indexOfEndOfFirstWord = textCompletion.indexOf(" ", textCompletion.startsWith(" ") ? 1 : 0);
        // first whole word in the textCompletion
        textCompletion = textCompletion.substring(0, indexOfEndOfFirstWord);
      } // [ CTRL + ENTER ] -> full textCompletion
      else if (key === ACTIONS.ENTER && ctrlKey) void 0;
      // [ ENTER ] -> first full line of textCompletion
      else if (key === ACTIONS.ENTER && !ctrlKey) {
        textCompletion = textCompletion.substring(0, textCompletion.length).split("\n")[0] + "\n";
        // if the key is not in the ACTIONS object, throw an error this would be a bug
      } else throw new Error(`key ${key} is not a valid ${ACTIONS}; refer to ACTIONS`);
      // dispatch PartialState
      fite.setPartialState({ textAreaValue: textPrompt + textCompletion });
    },
    [fite.textCompletion, textAreaRef]
  );

  const debouncedTextCompletionDispatch = React.useCallback(
    // debounce the generateText function to prevent spamming the server
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
          fite.setPartialState({ textCompletion: textPrompt.trim() + ".".repeat(count % 4) });
          // sleep for 250ms before dispatching the next textCompletion
          await new Promise((resolve) => setTimeout(resolve, 250));
          // increment count
          count++;
        }
        // dispatch the textCompletion with the results
        fite.setPartialState({ textCompletion: await results });
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
      if (fite.textCompletion === TAF_HEADER) return;
      if (!fite.textCompletion || !fite.textCompletion.startsWith(textAreaValue)) {
        fite.setPartialState({ textCompletion: textAreaValue });
        // make a call to the api using the lodash.debounce callback to dispatch the textCompletion
        debouncedTextCompletionDispatch(textAreaValue);
      }
      // always dispatch the textAreaValue
      fite.setPartialState({ textAreaValue });
    },
    [fite.textCompletion, fite.textAreaValue, debouncedTextCompletionDispatch, toUpperCase]
  );
  // return the InputLayer
  return (
    <textarea
      ref={textAreaRef}
      className={ClassNames.inputLayer}
      onKeyDown={handelTextAreaKeyDown}
      onChange={handleOnTextAreaChange}
      value={fite.textAreaValue}
    />
  );
}

export function AnnotationLayer({ children }: { children: React.ReactNode }): JSX.Element {
  // return the AnnotationLayer
  return (
    <div style={{ position: "absolute" }}>
      <pre className={ClassNames.annotationLayer}>{children}</pre>
    </div>
  );
}

export function TextCompletionLayer(): JSX.Element {
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
      .then((completion) => fite.setPartialState({ textCompletion: completion.join("\n") }));
  }, [fite.textAreaValue]);
  // return the TextAreaStack
  return <div className={ClassNames.textAreaStack}>{children}</div>;
}

export default function (): JSX.Element {
  /* a text area with intellisense autocomplete
    that is used for writing a terminal aerodrome forecast
    the intellisense is based the TAF_STRING above
    the user can cycle through the intellisense options with the up and down arrow keys
    the user can select an option by pressing tab or enter */

  return (
    <TextAreaStack>
      <TextCompletionLayer />
      <TextInputLayer />
    </TextAreaStack>
  );
}
