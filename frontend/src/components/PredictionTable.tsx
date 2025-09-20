import { FC } from 'react';
import {
  Box,
  Chip,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Typography
} from '@mui/material';
import { Prediction } from '../api';

interface PredictionTableProps {
  predictions: Prediction[];
  limit?: number;
}

const PredictionTable: FC<PredictionTableProps> = ({ predictions, limit = 50 }) => {
  if (!predictions.length) {
    return (
      <Paper variant="outlined" sx={{ p: 3 }}>
        <Typography variant="body2" color="text.secondary">
          No predictions available for this run.
        </Typography>
      </Paper>
    );
  }

  const displayed = predictions.slice(0, limit);

  const renderProbabilities = (metadata: Record<string, unknown>) => {
    const probabilities = metadata?.probabilities as Record<string, number> | undefined;
    if (!probabilities) {
      return null;
    }
    const sorted = Object.entries(probabilities).sort((a, b) => b[1] - a[1]).slice(0, 3);
    return (
      <Box display="flex" gap={1} flexWrap="wrap">
        {sorted.map(([label, value]) => (
          <Chip key={label} label={`${label}: ${(value * 100).toFixed(1)}%`} size="small" color="primary" variant="outlined" />
        ))}
      </Box>
    );
  };

  return (
    <Paper variant="outlined" sx={{ overflowX: 'auto' }}>
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>ID</TableCell>
            <TableCell>Text</TableCell>
            <TableCell>Expected</TableCell>
            <TableCell>Predicted</TableCell>
            <TableCell>Confidence</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {displayed.map((prediction) => {
            const text = (prediction.inputs?.text as string) ?? '';
            const isCorrect = prediction.expected_output === prediction.predicted_output;
            return (
              <TableRow key={prediction.uid} hover selected={!isCorrect}>
                <TableCell sx={{ fontWeight: 600 }}>{prediction.uid}</TableCell>
                <TableCell sx={{ maxWidth: 320 }}>
                  <Typography variant="body2" noWrap title={text}>
                    {text || '—'}
                  </Typography>
                </TableCell>
                <TableCell>{String(prediction.expected_output ?? '—')}</TableCell>
                <TableCell>
                  <Typography color={isCorrect ? 'success.main' : 'error.main'}>
                    {String(prediction.predicted_output ?? '—')}
                  </Typography>
                </TableCell>
                <TableCell>{renderProbabilities(prediction.metadata ?? {})}</TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </Paper>
  );
};

export default PredictionTable;
