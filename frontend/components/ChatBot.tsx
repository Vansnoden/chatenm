"use client";
import { useState } from "react";
import { X } from "lucide-react";
import { GET_IMAGE_URL, SEND_MSG_URL } from "@/app/lib/constants";

export default function ChatBot(){
    const [messages, setMessages] = useState<{ role: string; text: string; image: string }[]>([]);
    const [input, setInput] = useState("");
    const [visible, setVisible] = useState(true);
    const [thinking, setThinking] = useState(false);
    const [imageUrl, setImageURL] = useState("");
    const [tifUrl, setTifURL] = useState("");


    const sendMessage = async () => {
        if (!input.trim()) return;

        // Add user message
        const newMessages = [...messages, { role: "user", text: input, image: "" }];
        setMessages(newMessages);
        setInput("");

        //  bot response 
        try {
            setThinking(true);
            const response = await fetch(SEND_MSG_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                msg: input,
                session_id: 0
            }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const answer = await response.json();
            
            setMessages((prev) => [
                ...prev,
                { role: "assistant", text: answer.text, image: GET_IMAGE_URL + "/" + answer.image },
            ]);
            setImageURL(GET_IMAGE_URL + "/" + answer.image + ".png" );
            setTifURL(GET_IMAGE_URL + "/" + answer.image + ".tif" );
            setThinking(false);
        } catch (error) {
            console.error("Fetch error:", error);
        }
    };


    return(
        <div className="">
            { visible && (
                <div className="infoAlert flex items-center 
                    justify-between p-4 mb-4 rounded-xl bg-blue-50 border
                    border-blue-200 text-blue-700 shadow-sm">
                    <div className="text-sm">
                        <b className="">
                            This is an Example of input, you can use it as base and customize it to your needs.
                        </b> <br/><br/>
                        <p>
                            Download bioclim from worldclim with 10m resolution  <br/>
                            Download Elevation data with 10m resolution  <br/>
                            Download the Kenya shapefiles  <br/>
                            Download 100 records of Anopheles gambiae occurrences in Kenya from GBIF  <br/>
                            And run an ecological niche model using the downloaded elevation and occurrence data,  <br/>
                            considering bioclimatic variables 1 to 19. <br/>
                        </p>
                    </div>
                    <button onClick={() => setVisible(false)} 
                        className="ml-2 text-blue-600 hover:text-blue-800 focus:outline-none">
                        <X size={18} />
                    </button>
                </div>
            )}
            
            <div className="chatbotCtn flex flex-col mx-auto shadow-lg">
                {/* Messages */}
                <div className="flex-1 p-4 overflow-y-auto space-y-3">
                    {messages.map((msg, i) => (
                    <div
                        key={i}
                        className={`p-3 rounded-xl max-w-[55%] ${
                        msg.role === "user"
                            ? "ml-auto bg-blue-300 text-gray-900"
                            : "mr-auto bg-gray-200 text-gray-900"
                        }`}
                    >
                        {msg.text}
                        { msg.image  && <img className="chatImage" src={msg.image + '.png'} alt="image"></img> }
                    </div>
                    ))}
                </div>

                {/* Input */}
                <div className="border-t border-gray-200 p-3 flex gap-2">
                    <input
                        type="text"
                        placeholder="Type your message..."
                        className="flex-1 px-4 py-2 rounded-xl border border-gray-300 
                        focus:outline-none focus:ring-2 focus:ring-blue-500 
                        shadow-sm"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={(e) => e.key === "Enter" && sendMessage()}
                    />
                    <button
                        onClick={sendMessage}
                        className="px-4 py-2 rounded-xl bg-blue-600 text-white font-medium 
                        hover:bg-blue-700 active:bg-blue-800 
                        focus:outline-none focus:ring-2 focus:ring-blue-400 
                        transition"
                    >
                    Send
                    </button>
                </div>
                { thinking && (<span className="blink">Thinking ...</span>) }
                <span className="text-green-900 blink">{imageUrl}</span>
            </div>
        </div>
    )
}