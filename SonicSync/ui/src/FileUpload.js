import React, { useState } from "react";
import { IoMdCloudDone } from "react-icons/io";
import { Typography } from "@mui/material";
import RingLoader from "react-spinners/RingLoader";

const FileUpload = ({ setModels }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploaded, setUploaded] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleFileUpload = () => {
    const formData = new FormData();
    formData.append("file", selectedFile);
    setLoading(true);

    // Send the FormData to the backend using fetch or Axios
    fetch("/predict", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        // Handle the backend response
        console.log(data);
        setModels(data.result);
        setLoading(false);
        setUploaded(true);
        setTimeout(() => {
          setUploaded(false);
        }, 2000);
      })
      .catch((error) => {
        console.error(error);
      });
  };

  const override = {
    display: "block",
    position: "relative",
    top: "50%",
    left: "50%",
    transform: "translate(-75px, -75px)",
  };

  return (
    <>
      <div
        style={{
          backgroundColor: "white",
          paddingTop: "13px",
          paddingBottom: "13px",
          paddingLeft: "10px",
          paddingRight: "10px",
          borderRadius: "7px",
        }}
      >
        <input
          style={{
            paddingTop: "5px",
            paddingBottom: "5px",
            paddingLeft: "5px",
            paddingRight: "5px",
            borderRadius: "8px",
          }}
          type="file"
          onChange={handleFileChange}
        />
        <button
          style={{
            paddingTop: "10px",
            paddingBottom: "10px",
            paddingLeft: "20px",
            paddingRight: "20px",
            borderRadius: "8px",
            cursor: "pointer",
            border: "0",
            backgroundColor: "Highlight",
            color: "white",
            cursor: "pointer",
            fontWeight: "bold",
          }}
          onClick={handleFileUpload}
        >
          Upload
        </button>
      </div>
      {uploaded && (
        <div
          style={{
            position: "absolute",
            background: "white",
            top: "15%",
            width: "300px",
            display: "flex",
            alignItems: "center",
            gap: "10px",
            padding: "10px",
            height: "40px",
            borderRadius: "7px",
            transform: "translateX(-67%)",
            left: "50%",
          }}
        >
          <h1 style={{ color: "#355F2E" }}>
            <IoMdCloudDone />
          </h1>
          <Typography style={{ fontWeight: "bolder" }} variant="body2">
            Uploaded Succesfully
          </Typography>
        </div>
      )}
      {loading && (
        <div
          style={{
            width: "100%",
            height: "100vh",
            position: "absolute",
            top: "0%",
            left: "0%",
            backgroundColor: "rgba(0,0,0,0.5)",
            zIndex: 10,
          }}
        >
          <RingLoader
            color={"#fff"}
            loading={loading}
            size={150}
            aria-label="Loading Spinner"
            data-testid="loader"
            cssOverride={override}
          />
        </div>
      )}
    </>
  );
};

export default FileUpload;
