import { createTheme } from '@mui/material/styles';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2'
    },
    secondary: {
      main: '#9c27b0'
    },
    background: {
      default: '#f7f9fc'
    }
  },
  typography: {
    fontFamily: '"Inter", "Helvetica", "Arial", sans-serif'
  }
});

export default theme;
