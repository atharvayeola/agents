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

  const renderConfidenceCell = (metadata: Record<string, unknown>) => {
    const probabilities = metadata?.probabilities as Record<string, number> | undefined;
    const confidence = typeof metadata?.confidence === 'number' ? (metadata.confidence as number) : undefined;

    const elements: JSX.Element[] = [];
    if (typeof confidence === 'number') {
      elements.push(
        <Typography key="confidence" variant="body2">
          {(confidence * 100).toFixed(1)}%
        </Typography>
      );
    }

    if (probabilities) {
      const sorted = Object.entries(probabilities).sort((a, b) => b[1] - a[1]).slice(0, 3);
      elements.push(
        <Box key="probabilities" display="flex" gap={1} flexWrap="wrap">
          {sorted.map(([label, value]) => (
            <Chip key={label} label={`${label}: ${(value * 100).toFixed(1)}%`} size="small" color="primary" variant="outlined" />
          ))}
        </Box>
      );
    }

    if (!elements.length) {
      return <Typography variant="body2" color="text.secondary">—</Typography>;
    }

    return (
      <Box display="flex" flexDirection="column" gap={0.5} alignItems="flex-start">
        {elements}
      </Box>
    );
  };

  const renderRetrievedContexts = (metadata: Record<string, unknown>) => {
    const retrieved = metadata?.retrieved_documents as Array<Record<string, unknown>> | undefined;
    if (!Array.isArray(retrieved) || !retrieved.length) {
      return <Typography variant="body2" color="text.secondary">—</Typography>;
    }

    const top = retrieved.slice(0, 3);
    return (
      <Box display="flex" flexDirection="column" gap={0.5} alignItems="flex-start">
        {top.map((entry, index) => {
          const id = typeof entry?.id === 'string' ? entry.id : `doc-${index + 1}`;
          const score = typeof entry?.score === 'number' ? (entry.score as number) : undefined;
          const label = score !== undefined ? `${id} · ${(score * 100).toFixed(1)}%` : id;
          return <Chip key={`${id}-${index}`} label={label} size="small" variant="outlined" color="secondary" />;
        })}
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
            <TableCell>Retrieved Contexts</TableCell>
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
                <TableCell>{renderConfidenceCell(prediction.metadata ?? {})}</TableCell>
                <TableCell>{renderRetrievedContexts(prediction.metadata ?? {})}</TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </Paper>
  );
};

export default PredictionTable;
