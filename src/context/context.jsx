import { createContext, useState } from "react";
import axios from "axios";
import './context.css'
export const Context = createContext();

const ContextProvider = (props) => {
    const [input, setInput] = useState("");
    const [recentPrompt, setRecentPrompt] = useState("");
    const [prevPrompts, setPrevPrompts] = useState([]);
    const [showResult, setShowResult] = useState(false);
    const [loading, setLoading] = useState(false);
    const [resultData, setResultData] = useState("");

    const delayPara = (index, nextWord) => {
        setTimeout(function () {
            setResultData((prev) => prev + nextWord);
        }, 75 * index);
    };

    const newChat = () => {
        setLoading(false);
        setShowResult(false);
    };

    const onSent = async(prompt, files) => {
        setResultData("");
        setLoading(true);
        setShowResult(true);
    
        // Ensure files is an array (in case only one file is selected)
        const fileArray = Array.isArray(files) ? files : [files];
    
        let formData = new FormData();
        formData.append('message', prompt);
    
        // Append all files to the formData
        fileArray.forEach((file) => {
            formData.append('files', file);
        });
    
        try {
            const response = await axios.post("http://localhost:5000/process", formData, {
                headers: { "Content-Type": "multipart/form-data" },
            });
            const apiResponse = response.data.message;
            const responseArray = apiResponse.split("**");
            let newResponse = "";


            for (let i = 0; i < responseArray.length; i++) {
                if (i === 0 || i % 2 !== 1) {
                    newResponse += responseArray[i];
                } else {
                    newResponse += "<b>" + responseArray[i] + "</b>";
                }
            }
            let newResponse2 = newResponse.split("*").join("</br>");
            let newResponseArray = newResponse2.split(" ");
            for (let i = 0; i < newResponseArray.length; i++) {
                const nextWord = newResponseArray[i];
                delayPara(i, nextWord + " ");
            }
            console.log(apiResponse);
            // console.log(apiResponse);
            // const formattedResponse = formatResponse(apiResponse);
            // setResultData(formattedResponse);
        } catch (error) {
            console.error("Error Details:", error.response ? error.response.data : error.message);
            setResultData("Error: Unable to process your request.");
        }
    
        setLoading(false);
        setInput("");
    };
        
    const contextValue = {
        prevPrompts,
        setPrevPrompts,
        onSent,
        setRecentPrompt,
        recentPrompt,
        showResult,
        loading,
        resultData,
        input,
        setInput,
        newChat,
    };

    return <Context.Provider value={contextValue}>{props.children}</Context.Provider>;
};

export default ContextProvider;
