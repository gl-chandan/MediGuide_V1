import { useState, useEffect, useRef } from "react";
import "./MedicalChat.css";

function MedicalChat() {

  // ============================================================
  // STATES
  // ============================================================

  const [messages, setMessages] = useState([]);

  const [input, setInput] = useState("");

  const [chatHistory, setChatHistory] = useState(() => {

    const savedChats = localStorage.getItem(
      "mediguide_chats"
    );

    return savedChats
      ? JSON.parse(savedChats)
      : [];

  });

  const [currentChatId, setCurrentChatId] =
    useState(null);

  const messagesEndRef = useRef(null);


  // ============================================================
  // AUTO SCROLL
  // ============================================================

  useEffect(() => {

    messagesEndRef.current?.scrollIntoView({
      behavior: "smooth",
    });

  }, [messages]);


  // ============================================================
  // LOAD FIRST CHAT
  // ============================================================

  useEffect(() => {

    if (
      chatHistory.length > 0 &&
      currentChatId === null
    ) {

      setCurrentChatId(chatHistory[0].id);

      setMessages(chatHistory[0].messages);

    }

  }, []);


  // ============================================================
  // SAVE CHAT HISTORY
  // ============================================================

  useEffect(() => {

    if (!currentChatId) return;

    const updatedChats = chatHistory.map(
      (chat) =>

        chat.id === currentChatId
          ? {
              ...chat,
              messages: messages,
            }
          : chat
    );

    setChatHistory(updatedChats);

    localStorage.setItem(
      "mediguide_chats",
      JSON.stringify(updatedChats)
    );

  }, [messages]);


  // ============================================================
  // CREATE NEW CHAT
  // ============================================================

  function createNewChat() {

    const newChat = {

      id: Date.now(),

      title: "New Consultation",

      messages: [],

    };


    const updatedChats = [

      newChat,

      ...chatHistory,

    ];


    setChatHistory(updatedChats);

    localStorage.setItem(
      "mediguide_chats",
      JSON.stringify(updatedChats)
    );


    setCurrentChatId(newChat.id);

    setMessages([]);

  }


  // ============================================================
  // LOAD CHAT
  // ============================================================

  function loadChat(chat) {

    setCurrentChatId(chat.id);

    setMessages(chat.messages);

  }


  // ============================================================
  // DELETE CHAT
  // ============================================================

  function deleteChat(chatId) {

    const updatedChats = chatHistory.filter(
      (chat) => chat.id !== chatId
    );


    setChatHistory(updatedChats);

    localStorage.setItem(
      "mediguide_chats",
      JSON.stringify(updatedChats)
    );


    if (updatedChats.length > 0) {

      setCurrentChatId(
        updatedChats[0].id
      );

      setMessages(
        updatedChats[0].messages
      );

    } else {

      setCurrentChatId(null);

      setMessages([]);

    }

  }


  // ============================================================
  // STREAMING EFFECT
  // ============================================================

  async function streamText(fullText) {

    let currentText = "";


    for (
      let i = 0;
      i < fullText.length;
      i++
    ) {

      currentText += fullText[i];


      setMessages((prev) => {

        const updated = [...prev];

        updated[updated.length - 1] = {

          sender: "ai",

          text: currentText,

        };

        return updated;

      });


      await new Promise(
        (resolve) =>
          setTimeout(resolve, 20)
      );

    }

  }


  // ============================================================
  // SEND MESSAGE
  // ============================================================

  async function sendMessage() {

    if (input.trim() === "") return;


    let activeChatId = currentChatId;


    // ==========================================================
    // CREATE CHAT IF NONE EXISTS
    // ==========================================================

    if (!activeChatId) {

      const newChat = {

        id: Date.now(),

        title: input.slice(0, 30),

        messages: [],

      };


      const updatedChats = [

        newChat,

        ...chatHistory,

      ];


      setChatHistory(updatedChats);

      localStorage.setItem(
        "mediguide_chats",
        JSON.stringify(updatedChats)
      );


      setCurrentChatId(
        newChat.id
      );

      activeChatId = newChat.id;

    }


    // ==========================================================
    // USER MESSAGE
    // ==========================================================

    const userMessage = {

      sender: "user",

      text: input,

    };


    const currentInput = input;


    // ==========================================================
    // ADD USER + LOADING MESSAGE
    // ==========================================================

    setMessages((prev) => [

      ...prev,

      userMessage,

      {
        sender: "ai",
        text: "Analyzing...",
      },

    ]);


    setInput("");


    // ==========================================================
    // API REQUEST
    // ==========================================================

    try {

      const response = await fetch(
        "http://127.0.0.1:8000/predict",
        {

          method: "POST",

          headers: {

            "Content-Type":
              "application/json",

          },

          body: JSON.stringify({

            symptoms: currentInput,

          }),

        }
      );


      // ========================================================
      // CHECK HTTP RESPONSE
      // ========================================================

      if (!response.ok) {

        const errorData =
          await response.json();

        throw new Error(
          errorData.detail ||
          "Prediction failed"
        );

      }


      // ========================================================
      // GET API DATA
      // ========================================================

      const data =
        await response.json();


      // ========================================================
      // CONFIDENCE
      // ========================================================

      const confidence =
        (
          data.confidence * 100
        ).toFixed(2);


      // ========================================================
      // PROBABILITY DISTRIBUTION
      // ========================================================

      let probabilityText =
        "\n\nProbability Distribution:\n";


      Object.entries(
        data.probabilities
      )
        .sort(
          ([, a], [, b]) =>
            b - a
        )
        .forEach(
          ([specialty, probability]) => {

            probabilityText +=
              `${specialty}: ` +
              `${(
                probability * 100
              ).toFixed(2)}%\n`;

          }
        );


      // ========================================================
      // FINAL AI RESPONSE
      // ========================================================

      const aiResponse =

        `Predicted Specialty: ` +
        `${data.prediction}\n\n` +

        `Confidence: ` +
        `${confidence}%` +

        probabilityText;


      // ========================================================
      // STREAM RESPONSE
      // ========================================================

      await streamText(
        aiResponse
      );

    }


    // ==========================================================
    // ERROR HANDLING
    // ==========================================================

    catch (error) {

      console.error(
        "Prediction Error:",
        error
      );


      await streamText(
        "Unable to connect to the MediGuide server. Please make sure the backend is running."
      );

    }

  }


  // ============================================================
  // ENTER KEY SUPPORT
  // ============================================================

  function handleKeyDown(e) {

    if (
      e.key === "Enter" &&
      !e.shiftKey
    ) {

      e.preventDefault();

      sendMessage();

    }

  }


  // ============================================================
  // UI
  // ============================================================

  return (

    <div className="app-container">


      {/* ======================================================
          SIDEBAR
          ====================================================== */}

      <div className="sidebar">


        <div className="sidebar-header">

          MediGuide AI

        </div>


        <button
          className="new-chat-btn"
          onClick={createNewChat}
        >

          + New Chat

        </button>


        <div className="chat-history">

          {chatHistory.map(
            (chat) => (

              <div
                className="chat-item-wrapper"
                key={chat.id}
              >


                <div
                  className="chat-item"
                  onClick={() =>
                    loadChat(chat)
                  }
                >

                  {chat.title}

                </div>


                <button
                  className="delete-btn"
                  onClick={() =>
                    deleteChat(
                      chat.id
                    )
                  }
                >

                  X

                </button>


              </div>

            )
          )}

        </div>

      </div>


      {/* ======================================================
          CHAT SECTION
          ====================================================== */}

      <div className="chat-section">


        {/* ====================================================
            MESSAGES
            ==================================================== */}

        <div className="messages-area">

          {messages.map(
            (msg, index) => (

              <div
                key={index}
                className={
                  msg.sender === "user"
                    ? "message user-message"
                    : "message ai-message"
                }
              >

                {msg.text}

              </div>

            )
          )}


          <div
            ref={messagesEndRef}
          />

        </div>


        {/* ====================================================
            INPUT AREA
            ==================================================== */}

        <div className="input-area">


          <textarea

            placeholder="Describe your symptoms..."

            value={input}

            onChange={(e) =>
              setInput(
                e.target.value
              )
            }

            onKeyDown={
              handleKeyDown
            }

          />


          <button
            onClick={sendMessage}
          >

            Send

          </button>


        </div>

      </div>

    </div>

  );

}


export default MedicalChat;