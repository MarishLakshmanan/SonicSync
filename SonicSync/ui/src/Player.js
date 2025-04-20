import React, {useEffect, useRef, useState} from 'react';
import {Grid, IconButton, LinearProgress, Slider, Switch, Typography} from "@mui/material";
import {
  PlayArrow as PlayIcon,
  SkipNext as SkipNextIcon,
  SkipPrevious as SkipPreviousIcon,
  Pause as PauseIcon,
  VolumeUp as VolumeIcon
} from "@mui/icons-material";

const Player = (props) => {
  const [trackIx, setTrackIx] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const progressIntervalRef = useRef(null);
  const [progress, setProgress] = useState(0);
  const [playlist, setPlaylist] = useState([]);
  const storedGain = parseFloat(window.localStorage.getItem('gain'));
  const [gain, setGain] = useState(isNaN(storedGain) ? 50 : storedGain);

  const initSourceNode = (ix) => {
    if (playlist[ix].audio === null) {
      const newPlaylist = [ ...playlist ];
      newPlaylist[ix].audio = new Audio(newPlaylist[ix].url);
      newPlaylist[ix].audio.crossOrigin = 'anonymous';
      newPlaylist[ix].audio.loop = true;
      newPlaylist[ix].sourceNode = props.audioContext.createMediaElementSource(newPlaylist[ix].audio);
      newPlaylist[ix].sourceNode.connect(props.audioDestination);
      setPlaylist(newPlaylist);
    }
  };

  const skip = (newTrackIx) => {
    if (playlist.length && playlist[trackIx] && playlist[trackIx].audio) {
      playlist[trackIx].audio.pause();
    }
    initSourceNode(newTrackIx);
    playlist[newTrackIx].audio.currentTime = 0;
    setTrackIx(newTrackIx);
    setProgress(0);
    if (isPlaying) {
      playlist[newTrackIx].audio.play();
    }
  };

  const onSkipPreviousClick = () => {
    skip(trackIx > 0 ? trackIx - 1 : playlist.length - 1);
  };

  const onSkipNextClick = () => {
    skip(trackIx < playlist.length - 1 ? trackIx + 1 : 0);
  };

  const onPlayClick = () => {
    if (!playlist.length) return;
    initSourceNode(trackIx);
    if (isPlaying) {
      playlist[trackIx].audio.pause();
    } else {
      playlist[trackIx].audio.play();
    }
    setIsPlaying(!isPlaying);
  };

  useEffect(() => {
    if (!playlist.length) return;
    clearInterval(progressIntervalRef.current);
    progressIntervalRef.current = null;
    if (isPlaying) {
      progressIntervalRef.current = setInterval(() => {
        setProgress(playlist[trackIx].audio.currentTime / playlist[trackIx].audio.duration * 100);
      }, 10)
    }
    return () => {
      clearInterval(progressIntervalRef.current);
      progressIntervalRef.current = null;
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trackIx, isPlaying, playlist]);

  useEffect(() => {
    if (playlist.length) return;
    fetch('/playlist').then(async res => {
      const playlist = await res.json();
      for (let i = 0; i < playlist.length; ++i) {
        playlist[i].audio = null;
        playlist[i].sourceNode = null;
      }
      setPlaylist(playlist);
    });
  }, [playlist.length]);

  const onGainChange = (val) => {
    setGain(val);
    window.localStorage.setItem('gain', val);
    props.onGainChange(val);
  };

  if (!playlist.length) return null;

  return (
    <Grid
      container
      direction='column'
      justifyContent='center'
      alignItems='center'
      sx={{width: '100%', transition: 'transform 0.5s ease 0.2s'}}
      style={{transform: props.isEqEnabled ? 'translate(0, 0)' : 'translateY(200px)'}}
    >
      <Grid item sx={{width: '100%'}}>
        <div>
          <Grid
            item container direction='column' justifyContent='center' alignItems='center'
            sx={{
              background: 'rgba(255, 255, 255, 0.93)',
              borderRadius: {xs: 0, sm: 2},
              padding: 1,
              borderStyle: 'solid',
              borderWidth: 1,
              borderColor: theme => theme.palette.grey.A400,
              backdropFilter: 'blur(3px)'
            }}
          >
            <Grid item>
              <Typography variant='caption'>{playlist[trackIx].name}</Typography>
            </Grid>
            <Grid item>
              <Grid
                container direction='row' justifyContent='center' alignItems='center' columnSpacing={1}
              >
                <Grid item>
                  <IconButton onClick={onSkipPreviousClick} color='primary'>
                    <SkipPreviousIcon />
                  </IconButton>
                </Grid>
                <Grid item>
                  <IconButton onClick={onPlayClick} color='primary'>
                    {isPlaying ? <PauseIcon /> : <PlayIcon />}
                  </IconButton>
                </Grid>
                <Grid item>
                  <IconButton onClick={onSkipNextClick} color='primary'>
                    <SkipNextIcon />
                  </IconButton>
                </Grid>
                <Grid item sx={{width: {sm: '210px'}}}>
                  <LinearProgress
                    sx={{ '& .MuiLinearProgress-bar': { transition: 'none' } }}
                    variant='determinate'
                    color='primary'
                    value={progress}
                  />
                </Grid>
                <Grid
                  item container direction='row' justifyContent='center' alignItems='center'
                  sx={{width: '120px', display: {xs: 'none', sm: 'flex'}}}
                >
                  <Grid item>
                    <IconButton color='primary'>
                      <VolumeIcon />
                    </IconButton>
                  </Grid>
                  <Grid item sx={{flexGrow: 1}}>
                    <Slider value={gain} onChange={(e, val) => { onGainChange(val); }} size='medium' />
                  </Grid>
                </Grid>
                <Grid
                  item container direction='row' justifyContent='center' alignItems='center'
                  sx={{width: '90px'}}
                >
                  <Grid item>
                    <Typography>EQ</Typography>
                  </Grid>
                  <Grid item>
                    <Switch
                      checked={props.isEqOn}
                      onChange={(e, val) => { props.onIsEqOnChange(val); }}
                      disabled={!props.isEqEnabled}
                    />
                  </Grid>
                </Grid>
              </Grid>
            </Grid>
          </Grid>
        </div>
      </Grid>
    </Grid>
  );
};

export default Player;
