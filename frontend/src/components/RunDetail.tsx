import { FC, useMemo } from 'react';
import {
  Box,
  CircularProgress,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Typography
} from '@mui/material';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from 'recharts';
import { Metric, RunDetail as RunDetailType } from '../api';
import MetricsCards from './MetricsCards';
import PredictionTable from './PredictionTable';

interface RunDetailProps {
  run: RunDetailType | null;
  loading: boolean;
}

const getMetricByName = (metrics: Metric[], name: string) =>
  metrics.find((metric) => metric.name.toLowerCase() === name.toLowerCase());

const RunDetail: FC<RunDetailProps> = ({ run, loading }) => {
  const labelDistribution = useMemo(() => {
    if (!run) return null;
    const metric = getMetricByName(run.metrics, 'label_distribution');
    if (!metric) return null;
    const gold = metric.details?.gold as Record<string, number> | undefined;
    const predicted = metric.details?.predicted as Record<string, number> | undefined;
    if (!gold || !predicted) return null;
    const labels = Array.from(new Set([...Object.keys(gold), ...Object.keys(predicted)]));
    return labels.map((label) => ({
      label,
      gold: gold[label] ?? 0,
      predicted: predicted[label] ?? 0
    }));
  }, [run]);

  const confusionMatrix = useMemo(() => {
    if (!run) return null;
    const metric = getMetricByName(run.metrics, 'confusion_matrix');
    if (!metric) return null;
    const labels = (metric.details?.labels as string[]) ?? [];
    const matrix = (metric.details?.matrix as number[][]) ?? [];
    return { labels, matrix };
  }, [run]);

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="100%">
        <CircularProgress />
      </Box>
    );
  }

  if (!run) {
    return (
      <Paper variant="outlined" sx={{ p: 4, textAlign: 'center' }}>
        <Typography variant="body1" color="text.secondary">
          Select a run from the left to inspect detailed metrics and predictions.
        </Typography>
      </Paper>
    );
  }

  return (
    <Box display="flex" flexDirection="column" gap={3}>
      <Paper variant="outlined" sx={{ p: 3 }}>
        <Typography variant="h5" fontWeight={600} gutterBottom>
          {run.name}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Config: {run.config_name} · Task: {run.task}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Started {new Date(run.started_at).toLocaleString()} · Completed {new Date(run.completed_at).toLocaleString()} ·
          Duration {run.duration.toFixed(2)}s
        </Typography>
      </Paper>

      <MetricsCards metrics={run.metrics} />

      {labelDistribution && labelDistribution.length > 0 && (
        <Paper variant="outlined" sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Label distribution
          </Typography>
          <Box sx={{ width: '100%', height: 300 }}>
            <ResponsiveContainer>
              <BarChart data={labelDistribution} margin={{ top: 16, right: 24, bottom: 0, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="label" />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Legend />
                <Bar dataKey="gold" fill="#1976d2" name="Gold" />
                <Bar dataKey="predicted" fill="#9c27b0" name="Predicted" />
              </BarChart>
            </ResponsiveContainer>
          </Box>
        </Paper>
      )}

      {confusionMatrix && confusionMatrix.labels.length > 0 && (
        <Paper variant="outlined" sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Confusion matrix
          </Typography>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Gold \ Predicted</TableCell>
                {confusionMatrix.labels.map((label) => (
                  <TableCell key={label}>{label}</TableCell>
                ))}
              </TableRow>
            </TableHead>
            <TableBody>
              {confusionMatrix.matrix.map((row, rowIndex) => (
                <TableRow key={confusionMatrix.labels[rowIndex]}>
                  <TableCell component="th" scope="row" sx={{ fontWeight: 600 }}>
                    {confusionMatrix.labels[rowIndex]}
                  </TableCell>
                  {row.map((value, columnIndex) => (
                    <TableCell key={`${rowIndex}-${columnIndex}`}>{value}</TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </Paper>
      )}

      <Box>
        <Typography variant="h6" gutterBottom>
          Predictions
        </Typography>
        <PredictionTable predictions={run.predictions} />
      </Box>
    </Box>
  );
};

export default RunDetail;
