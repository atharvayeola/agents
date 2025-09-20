import { FC } from 'react';
import { Grid, Paper, Typography } from '@mui/material';
import { Metric } from '../api';

interface MetricsCardsProps {
  metrics: Metric[];
}

const shouldDisplay = (metric: Metric) => {
  const name = metric.name.toLowerCase();
  return !(name.includes('distribution') || name.includes('confusion'));
};

const formatMetricValue = (value: number) => {
  if (Number.isNaN(value)) {
    return '—';
  }
  if (value <= 1) {
    return `${(value * 100).toFixed(1)}%`;
  }
  return value.toFixed(3);
};

const prettifyName = (name: string) =>
  name
    .replace(/[_-]/g, ' ')
    .split(' ')
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');

const MetricsCards: FC<MetricsCardsProps> = ({ metrics }) => {
  const displayMetrics = metrics.filter(shouldDisplay);
  if (!displayMetrics.length) {
    return null;
  }

  return (
    <Grid container spacing={2} sx={{ mb: 2 }}>
      {displayMetrics.map((metric) => (
        <Grid item xs={12} sm={6} md={3} key={metric.name}>
          <Paper variant="outlined" sx={{ p: 2 }}>
            <Typography variant="subtitle2" color="text.secondary">
              {prettifyName(metric.name)}
            </Typography>
            <Typography variant="h4" fontWeight={600} mt={1}>
              {formatMetricValue(metric.value)}
            </Typography>
          </Paper>
        </Grid>
      ))}
    </Grid>
  );
};

export default MetricsCards;
