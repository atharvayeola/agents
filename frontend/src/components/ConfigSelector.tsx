import { FC } from 'react';
import {
  Box,
  Button,
  FormControl,
  FormControlLabel,
  InputLabel,
  MenuItem,
  Select,
  Switch,
  Typography
} from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import { ConfigInfo } from '../api';

interface ConfigSelectorProps {
  configs: ConfigInfo[];
  selectedConfig: string;
  onSelect: (config: string) => void;
  onTrigger: () => void;
  triggering: boolean;
  savePredictions: boolean;
  onSavePredictionsChange: (value: boolean) => void;
}

const ConfigSelector: FC<ConfigSelectorProps> = ({
  configs,
  selectedConfig,
  onSelect,
  onTrigger,
  triggering,
  savePredictions,
  onSavePredictionsChange
}) => {
  return (
    <Box display="flex" alignItems="center" gap={2} flexWrap="wrap">
      <FormControl sx={{ minWidth: 220 }} size="small">
        <InputLabel id="config-select-label">Configuration</InputLabel>
        <Select
          labelId="config-select-label"
          label="Configuration"
          value={selectedConfig}
          onChange={(event) => onSelect(event.target.value)}
        >
          {configs.map((config) => (
            <MenuItem key={config.name} value={config.name}>
              <Box>
                <Typography variant="body1" fontWeight={600}>
                  {config.name}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {config.path}
                </Typography>
              </Box>
            </MenuItem>
          ))}
        </Select>
      </FormControl>

      <FormControlLabel
        control={
          <Switch
            checked={savePredictions}
            onChange={(event) => onSavePredictionsChange(event.target.checked)}
            size="small"
          />
        }
        label="Save predictions"
      />

      <Button
        variant="contained"
        startIcon={<PlayArrowIcon />}
        onClick={onTrigger}
        disabled={!selectedConfig || triggering}
      >
        {triggering ? 'Running…' : 'Run Evaluation'}
      </Button>
    </Box>
  );
};

export default ConfigSelector;
