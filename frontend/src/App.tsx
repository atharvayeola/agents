import { useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Box,
  Container,
  Grid,
  Snackbar,
  Stack,
  Typography
} from '@mui/material';
import ConfigSelector from './components/ConfigSelector';
import RunDetail from './components/RunDetail';
import RunList from './components/RunList';
import {
  ConfigInfo,
  RunDetail as RunDetailType,
  RunSummary,
  fetchConfigs,
  fetchRun,
  fetchRuns,
  triggerRun
} from './api';

const App = () => {
  const [configs, setConfigs] = useState<ConfigInfo[]>([]);
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [runsLoading, setRunsLoading] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [selectedRun, setSelectedRun] = useState<RunDetailType | null>(null);
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null);
  const [selectedConfig, setSelectedConfig] = useState<string>('');
  const [triggering, setTriggering] = useState(false);
  const [savePredictions, setSavePredictions] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  useEffect(() => {
    const initialize = async () => {
      try {
        const [configData, runsData] = await Promise.all([fetchConfigs(), fetchRuns()]);
        setConfigs(configData);
        setRuns(runsData);
        if (!selectedConfig && configData.length > 0) {
          setSelectedConfig(configData[0].name);
        }
        if (runsData.length > 0) {
          const runId = runsData[0].id;
          setSelectedRunId(runId);
          loadRunDetail(runId);
        }
      } catch (err) {
        setError('Failed to initialize dashboard. Ensure the API is reachable.');
      }
    };
    initialize();
  }, []);

  const loadRuns = async () => {
    setRunsLoading(true);
    try {
      const runData = await fetchRuns();
      setRuns(runData);
    } catch (err) {
      setError('Failed to fetch evaluation runs.');
    } finally {
      setRunsLoading(false);
    }
  };

  const loadRunDetail = async (runId: number) => {
    setDetailLoading(true);
    try {
      const detail = await fetchRun(runId);
      setSelectedRun(detail);
      setSelectedRunId(detail.id);
    } catch (err) {
      setError(`Failed to load run ${runId}.`);
    } finally {
      setDetailLoading(false);
    }
  };

  const handleTriggerRun = async () => {
    if (!selectedConfig) return;
    setTriggering(true);
    try {
      const detail = await triggerRun(selectedConfig, savePredictions);
      setSuccess(`Run '${detail.name}' completed successfully.`);
      setSelectedRun(detail);
      setSelectedRunId(detail.id);
      await loadRuns();
    } catch (err) {
      setError('Failed to execute evaluation. Check server logs for more details.');
    } finally {
      setTriggering(false);
    }
  };

  const handleRunSelection = (runId: number) => {
    loadRunDetail(runId);
  };

  const selectedConfigInfo = useMemo(
    () => configs.find((config) => config.name === selectedConfig),
    [configs, selectedConfig]
  );

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      <Stack spacing={3}>
        <Box display="flex" justifyContent="space-between" alignItems="center" flexWrap="wrap" gap={2}>
          <Box>
            <Typography variant="h4" fontWeight={700} gutterBottom>
              Evaluation Dashboard
            </Typography>
            {selectedConfigInfo && (
              <Typography variant="body2" color="text.secondary">
                API base: {import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000/api'}
              </Typography>
            )}
          </Box>
          <ConfigSelector
            configs={configs}
            selectedConfig={selectedConfig}
            onSelect={setSelectedConfig}
            onTrigger={handleTriggerRun}
            triggering={triggering}
            savePredictions={savePredictions}
            onSavePredictionsChange={setSavePredictions}
          />
        </Box>

        <Grid container spacing={3} alignItems="stretch">
          <Grid item xs={12} md={4}>
            <RunList
              runs={runs}
              selectedRunId={selectedRunId}
              onSelect={handleRunSelection}
              loading={runsLoading}
            />
          </Grid>
          <Grid item xs={12} md={8}>
            <RunDetail run={selectedRun} loading={detailLoading} />
          </Grid>
        </Grid>
      </Stack>

      <Snackbar
        open={Boolean(error)}
        autoHideDuration={6000}
        onClose={() => setError(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert severity="error" onClose={() => setError(null)} sx={{ width: '100%' }}>
          {error}
        </Alert>
      </Snackbar>

      <Snackbar
        open={Boolean(success)}
        autoHideDuration={4000}
        onClose={() => setSuccess(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert severity="success" onClose={() => setSuccess(null)} sx={{ width: '100%' }}>
          {success}
        </Alert>
      </Snackbar>
    </Container>
  );
};

export default App;
