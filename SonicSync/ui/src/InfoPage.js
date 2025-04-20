import { Container, Grid, IconButton, Typography, Box } from "@mui/material";
import ClearIcon from "@mui/icons-material/Clear";
import FileOpenOutlinedIcon from "@mui/icons-material/FileOpenOutlined";
import React from "react";
import {
  android as isAndroid,
  ios as isIos,
  linux as isLinux,
  macos as isMacos,
  windows as isWindows,
} from "platform-detect";
import LooksOneOutlinedIcon from "@mui/icons-material/LooksOneOutlined";
import LooksTwoOutlinedIcon from "@mui/icons-material/LooksTwoOutlined";
import Looks3OutlinedIcon from "@mui/icons-material/Looks3Outlined";
import Looks4OutlinedIcon from "@mui/icons-material/Looks4Outlined";
import { useTheme } from "@emotion/react";
import Link from "@mui/material/Link";
import Lottie from "lottie-react-web";
import animationData from "./anima.json";

const InfoPage = (props) => {
  const platform = isWindows
    ? "Windows"
    : isMacos
    ? "Mac OS"
    : isLinux
    ? "Linux"
    : isAndroid
    ? "Android"
    : isIos
    ? "OSX"
    : null;
  const recommendedApp = {
    Windows: "EqualizerAPO GraphicEq",
    "Mac OS": "Sound Source",
    Linux: "EasyEffects",
    Android: "Wavelet",
    iOS: null,
  }[platform];

  const theme = useTheme();
  const iconSize = 48;

  return (
    <Container
      fixed
      maxWidth="xl"
      sx={{ color: (theme) => theme.palette.grey.A200, pb: 4 }}
      style={{ display: "flex", flexDirection: "row", marginTop: "140px" }}
    >
      <div>
        <Grid
          container
          direction="column-reverse"
          alignItems="center"
          style={{ display: "flex" }}
          rowSpacing={{ xs: 2, sm: 6 }}
          sx={{
            "& p": { pb: (theme) => theme.spacing(1) },
            pt: { xs: 4, sm: 8 },
          }}
        >
          <Grid
            item
            direction="column"
            sx={{ textAlign: "center" }}
            style={{ pt: { xs: 4, sm: 8 }, pb: { xs: 2, sm: 4 } }}
          >
            <Box
              sx={{
                width: { xs: "60vw", sm: 300, xl: 400 },
                maxWidth: { xs: 300, sm: "60vw" },
              }}
            >
              <img src="/SonicSync.svg" alt="logo" style={{ width: "150%" }} />
              {/* <Typography variant="body2">
              Make your headphones sound better
            </Typography> */}
            </Box>

            <Grid
              item
              container
              direction="column"
              columnSpacing={4}
              rowSpacing={4}
              justifyContent="center"
            >
              <Grid
                item
                xs={12}
                sm={6}
                container
                direction="row"
                alignItems="start"
                sx={{ textAlign: "left" }}
              >
                <Grid item sx={{ width: iconSize + 8 }}>
                  <LooksOneOutlinedIcon
                    sx={{ width: iconSize, height: iconSize }}
                  />
                </Grid>
                <Grid item sx={{ width: `calc(100% - ${iconSize + 8}px)` }}>
                  <Typography variant="h6" sx={{ lineHeight: 1.2, mb: "12px" }}>
                    Upload your music
                  </Typography>
                  <Typography variant="body2">
                    Before you select your headphone upload your music to get a
                    personlaized equalier for your headphone
                  </Typography>
                </Grid>
              </Grid>
              <Grid
                item
                xs={12}
                sm={6}
                container
                direction="row"
                alignItems="start"
                sx={{ textAlign: "left" }}
              >
                <Grid item sx={{ width: iconSize + 8 }}>
                  <LooksTwoOutlinedIcon
                    sx={{ width: iconSize, height: iconSize }}
                  />
                </Grid>
                <Grid item sx={{ width: `calc(100% - ${iconSize + 8}px)` }}>
                  <Typography variant="h6" sx={{ lineHeight: 1.2, mb: "12px" }}>
                    Select your headphones at the top
                  </Typography>
                  <Typography variant="body2">
                    In case your headphone is not listed you can use a differet
                    model from the same brand it should be good
                  </Typography>
                </Grid>
              </Grid>

              <Grid
                item
                xs={12}
                sm={6}
                container
                direction="row"
                alignItems="start"
                sx={{ textAlign: "left" }}
              >
                <Grid item sx={{ width: iconSize + 8 }}>
                  <Looks3OutlinedIcon
                    sx={{ width: iconSize, height: iconSize }}
                  />
                </Grid>
                <Grid item sx={{ width: `calc(100% - ${iconSize + 8}px)` }}>
                  <Typography variant="h6" sx={{ lineHeight: 1.2, mb: "12px" }}>
                    Get your equalizer
                  </Typography>
                  <Typography variant="body2">
                    SonicSync doesn't do the live equalization for your device
                    but it does give you the settings. select the equalizer app
                    you want to use and then you can manually enter the value to
                    the equalizer or import the file to the equalizer app.
                  </Typography>
                </Grid>
              </Grid>
            </Grid>
          </Grid>
        </Grid>
      </div>
      <div>
        <Grid>
          <Lottie
            style={{ width: "50%" }}
            options={{
              animationData: animationData,
              loop: true,
              autoplay: true,
            }}
          />
        </Grid>
      </div>
    </Container>
  );
};

export default InfoPage;
