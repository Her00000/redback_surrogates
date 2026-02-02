from collections import namedtuple
from redback_surrogates.utils import citation_wrapper
from joblib import load
import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import astropy.units as uu
from functools import lru_cache
import os
import h5py
import torch
import torch.nn as nn
dirname = os.path.dirname(__file__)
data_folder = os.path.join(dirname, "surrogate_data")

@citation_wrapper("https://ui.adsabs.harvard.edu/abs/2025arXiv250602107S/abstract, https://ui.adsabs.harvard.edu/abs/2023PASJ...75..634M/abstract")
class EnhancedSpectralModel:
    def __init__(self, latent_dim=64, learning_rate=1e-3, use_pca=True, pca_components=32):
        """Initialize the enhanced spectral model with optimized parameters

        Args:
            latent_dim: Dimension of latent space (reduced from 256 to 64)
            learning_rate: Learning rate for model training
            use_pca: Whether to use PCA for dimensionality reduction
            pca_components: Number of PCA components if use_pca is True
        """
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.encoder = None
        self.decoder = None
        self.regressor = None
        self.param_scaler = None
        self.flux_scaler = None
        self.latent_scaler = None
        self.standard_times = None
        self.standard_freqs = None
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.pca = None

    def predict_spectrum(self, parameters):
        """Predict spectrum for given parameters

        Args:
            parameters: DataFrame or array of physical parameters

        Returns:
            Predicted spectrum (time_dim, freq_dim)
        """
        # Convert to numpy array if DataFrame
        if isinstance(parameters, pd.DataFrame):
            param_array = parameters.values
        else:
            param_array = np.atleast_2d(parameters)

        # Scale parameters
        param_scaled = self.param_scaler.transform(param_array)

        # Predict latent representation (scaled)
        scaled_latent = self.regressor.predict(param_scaled, verbose=0)

        if self.use_pca and self.pca is not None:
            # Inverse scale the reduced latent space
            reduced_latent = self.inverse_scale_latent(scaled_latent)

            # Inverse transform to full latent space
            latent = self.pca.inverse_transform(reduced_latent)
        else:
            # Direct inverse scaling of latent space
            latent = self.inverse_scale_latent(scaled_latent)

        # Decode to scaled spectrum
        scaled_spectrum = self.decoder.predict(latent, verbose=0)

        # Inverse scale to original flux range
        spectrum = self.inverse_preprocess_flux(scaled_spectrum)

        # Return first spectrum if only one set of parameters
        if param_array.shape[0] == 1:
            return spectrum[0]

        return spectrum

    def inverse_preprocess_flux(self, scaled_flux):
        """Convert scaled flux back to original scale

        Args:
            scaled_flux: Scaled flux arrays

        Returns:
            Original scale flux arrays
        """
        if self.flux_scaler is None:
            print("Warning: flux_scaler not found, returning unscaled data")
            return scaled_flux

        # Reshape to 2D for inverse scaling
        orig_shape = scaled_flux.shape
        flux_2d = scaled_flux.reshape(orig_shape[0], -1)

        # Apply inverse scaling
        orig_2d = self.flux_scaler.inverse_transform(flux_2d)

        # Reshape back to original shape
        orig_flux = orig_2d.reshape(orig_shape)

        return orig_flux

    def inverse_scale_latent(self, scaled_latent):
        """Convert scaled latent values back to original scale

        Args:
            scaled_latent: Scaled latent values

        Returns:
            Original scale latent values
        """
        if self.latent_scaler is None:
            print("Warning: latent_scaler not found, returning unscaled data")
            return scaled_latent

        # Apply inverse scaling
        original_latent = self.latent_scaler.inverse_transform(scaled_latent)

        return original_latent

    @classmethod
    def load_model(cls, directory=data_folder + '/TypeII_Moriya'):
        """Load saved model from disk

        Args:
            directory: Directory containing saved model

        Returns:
            EnhancedSpectralModel instance with loaded models
        """
        # Load configuration
        import json
        with open(os.path.join(directory, 'config.json'), 'r') as f:
            config = json.load(f)

        # Initialize model with loaded config
        model = cls(
            latent_dim=config['latent_dim'],
            use_pca=config['use_pca'],
            pca_components=config['pca_components']
        )

        # Load encoder and decoder
        model.encoder = tf.keras.models.load_model(os.path.join(directory, 'encoder.keras'))
        model.decoder = tf.keras.models.load_model(os.path.join(directory, 'decoder.keras'))

        # Load regressor
        model.regressor = tf.keras.models.load_model(os.path.join(directory, 'regressor.keras'))

        # Load scalers
        import joblib
        model.param_scaler = joblib.load(os.path.join(directory, 'param_scaler.pkl'))
        model.flux_scaler = joblib.load(os.path.join(directory, 'flux_scaler.pkl'))

        # Load latent scaler if exists
        latent_scaler_path = os.path.join(directory, 'latent_scaler.pkl')
        if os.path.exists(latent_scaler_path):
            model.latent_scaler = joblib.load(latent_scaler_path)

        # Load PCA if exists
        pca_path = os.path.join(directory, 'pca.pkl')
        if os.path.exists(pca_path):
            model.pca = joblib.load(pca_path)

        # Load grid information
        grids = np.load(os.path.join(directory, 'standard_grids.npz'))
        model.standard_times = grids['times']
        model.standard_freqs = grids['freqs']

        return model

# Cached model loaders
@lru_cache(maxsize=1)
def _load_lbol_models():
    """Load and cache the lbol models and scalers."""
    lbolscaler = load(data_folder + '/TypeII_Moriya/lbolscaler.save')
    lbol_model = keras.saving.load_model(data_folder + '/TypeII_Moriya/lbol_model.keras')
    xscaler = load(data_folder + '/TypeII_Moriya/xscaler.save')
    return lbolscaler, lbol_model, xscaler

@lru_cache(maxsize=1)
def _load_photosphere_models():
    """Load and cache the photosphere models and scalers."""
    xscaler = load(data_folder + '/TypeII_Moriya/xscaler.save')
    tempscaler = load(data_folder + '/TypeII_Moriya/temperature_scaler.save')
    radscaler = load(data_folder + '/TypeII_Moriya/radius_scaler.save')
    temp_model = keras.saving.load_model(data_folder + '/TypeII_Moriya/temp_model.keras')
    rad_model = keras.saving.load_model(data_folder + '/TypeII_Moriya/radius_model.keras')
    return xscaler, tempscaler, radscaler, temp_model, rad_model

@lru_cache(maxsize=1)
def _load_spectra_model():
    """Load and cache the spectra model."""
    return EnhancedSpectralModel.load_model()

# Optional: Function to clear all caches if needed
def clear_typeII_model_cache():
    """Clear all cached models to free memory."""
    _load_lbol_models.cache_clear()
    _load_photosphere_models.cache_clear()
    _load_spectra_model.cache_clear()


# Updated functions using cached models
@citation_wrapper("https://ui.adsabs.harvard.edu/abs/2025arXiv250602107S/abstract, https://ui.adsabs.harvard.edu/abs/2023PASJ...75..634M/abstract")
def typeII_lbol(progenitor, ni_mass, log10_mdot, beta, rcsm, esn, **kwargs):
    """
    Generate bolometric light curve for Type II supernovae based on physical parameters (vectorised)

    :param progenitor: in solar masses
    :param ni_mass: in solar masses
    :param log10_mdot: in solar masses per year
    :param beta: dimensionless
    :param rcsm: in 10^14 cm
    :param esn: in 10^51
    :param kwargs: None
    :return: tts (in days in source frame) and bolometric luminosity (in erg/s)
    """
    rcsm = rcsm * 1e14
    log10_mdot = np.abs(log10_mdot)

    # Load cached models
    lbolscaler, lbol_model, xscaler = _load_lbol_models()

    tts = np.geomspace(1e-1, 400, 200)
    ss = np.array([progenitor, ni_mass, log10_mdot, beta, rcsm, esn]).T
    if isinstance(progenitor, float):
        ss = ss.reshape(1, -1)
    ss = xscaler.transform(ss)
    lbols = lbol_model(ss)
    lbols = lbolscaler.inverse_transform(lbols)
    if isinstance(progenitor, float):
        lbols = lbols.flatten()
    return tts, 10 ** lbols


@citation_wrapper("https://ui.adsabs.harvard.edu/abs/2025arXiv250602107S/abstract, https://ui.adsabs.harvard.edu/abs/2023PASJ...75..634M/abstract")
def typeII_photosphere(progenitor, ni_mass, log10_mdot, beta, rcsm, esn, **kwargs):
    """
    Generate synthetic photospheric temperature and radius for Type II supernovae based on physical parameters.
    (vectorised)

    :param progenitor: in solar masses
    :param ni_mass: in solar masses
    :param log10_mdot: in solar masses per year
    :param beta: dimensionless
    :param rcsm: in 10^14 cm
    :param esn: in 10^51
    :param kwargs: None
    :return: tts (in days in source frame) and temp (in K) and radius (in cm)
    """
    rcsm = rcsm * 1e14
    log10_mdot = np.abs(log10_mdot)

    # Load cached models
    xscaler, tempscaler, radscaler, temp_model, rad_model = _load_photosphere_models()

    tts = np.geomspace(1e-1, 400, 200)
    ss = np.array([progenitor, ni_mass, log10_mdot, beta, rcsm, esn]).T
    if isinstance(progenitor, float):
        ss = ss.reshape(1, -1)
    ss = xscaler.transform(ss)
    temp = temp_model(ss)
    rad = rad_model(ss)
    temp = tempscaler.inverse_transform(temp)
    rad = radscaler.inverse_transform(rad)
    if isinstance(progenitor, float):
        temp = temp.flatten()
        rad = rad.flatten()
    return tts, temp, rad


@citation_wrapper("https://ui.adsabs.harvard.edu/abs/2025arXiv250602107S/abstract, https://ui.adsabs.harvard.edu/abs/2023PASJ...75..634M/abstract")
def typeII_spectra(progenitor, ni_mass, log10_mdot, beta, rcsm, esn, **kwargs):
    """
    Generate synthetic spectra for Type II supernovae based on physical parameters.

    :param progenitor: in solar masses
    :param ni_mass: in solar masses
    :param log10_mdot: in solar masses per year
    :param beta: dimensionless
    :param rcsm: in 10^14 cm
    :param esn: in 10^51
    :param kwargs: None
    :return: rest-frame spectrum (luminosity density), frequency (in Angstrom) and time arrays and times (in days in source frame)
    """
    rcsm = rcsm * 1e14
    log10_mdot = np.abs(log10_mdot)
    # Create standard grids
    standard_times = np.geomspace(0.1, 400, 50)
    standard_freqs = np.geomspace(500, 49500, 50)

    # Create parameter dataframe for the surrogate model
    new_params = pd.DataFrame([{
        'progenitor': progenitor,
        'ni_mass': ni_mass,
        'mass_loss': log10_mdot,
        'beta': beta,
        'csm_radius': rcsm,
        'explosion_energy': esn
    }])

    # Get cached model
    model = _load_spectra_model()

    predicted_spectrum = model.predict_spectrum(new_params)
    # Apply empirical correction factor (if needed)
    predicted_spectrum = 10 ** predicted_spectrum
    # Convert to physical units (erg/s/Hz)
    rest_spectrum = predicted_spectrum * uu.erg / uu.s / uu.Hz

    output = namedtuple('output', ['spectrum', 'frequency', 'time'])
    return output(
        spectrum=rest_spectrum,
        frequency=standard_freqs * uu.Angstrom,
        time=standard_times * uu.day
    )


# ========== LOWME (Latent-space Optimized Waveform Emulator) Model ==========

class _LowmeResBlock(nn.Module):
    """Pre-activation ResNet Block with SiLU for LOWME"""
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        h = torch.nn.functional.silu(self.norm1(x))
        h = self.fc1(h)
        h = torch.nn.functional.silu(self.norm2(h))
        h = self.fc2(h)
        return x + h


class _LowmePhysicsEmbedding(nn.Module):
    """Physics parameter embedding for LOWME"""
    def forward(self, x):
        # Ensure 2D input
        if x.dim() == 1:
            x = x.unsqueeze(0)
        feats = [x]
        feats.append(x ** 2)
        feats.append(torch.tanh(x))
        feats.append(torch.log(torch.abs(x) + 1.1))
        if x.shape[1] >= 2:
            feats.append(x[:, 0:1] * x[:, 1:2])
        return torch.cat(feats, dim=1)


class LowmeLatentEmulator(nn.Module):
    """LOWME Latent Emulator: predicts latent codes from physical parameters"""
    def __init__(self, input_dim, latent_dim, hidden_dim=1024, num_blocks=12):
        super().__init__()
        
        self.register_buffer('x_mean', torch.zeros(input_dim))
        self.register_buffer('x_std', torch.ones(input_dim))
        
        self.physics_embed = _LowmePhysicsEmbedding()
        embed_dim = input_dim * 4 + 1
        
        self.input_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        
        self.blocks = nn.ModuleList([_LowmeResBlock(hidden_dim) for _ in range(num_blocks)])
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        
        self.latent_dim = latent_dim
    
    def forward(self, x):
        x = (x - self.x_mean) / self.x_std
        x = self.physics_embed(x)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)


class LowmeResNetDecoder(nn.Module):
    """LOWME ResNet Decoder: decodes latent codes to spectra"""
    def __init__(self, latent_dim, hidden_dim, output_dim, num_blocks):
        super().__init__()
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.blocks = nn.ModuleList([_LowmeResBlock(hidden_dim) for _ in range(num_blocks)])
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, z):
        h = self.input_proj(z)
        for block in self.blocks:
            h = block(h)
        h = self.output_norm(h)
        return self.output_proj(h)


@lru_cache(maxsize=1)
def _load_typeII_lowme_torch_bundle():
    """Load and cache the LOWME (Latent-space Optimized Waveform Emulator) torch surrogate.
    
    LOWME uses a two-stage architecture:
    - Emulator: physical params -> latent code (normalized)
    - Decoder: latent code -> spectrum (normalized [0,1])
    """
    directory = os.environ.get(
        'STELLA_LOWME_DIR',
        os.path.join(data_folder, 'TypeII_Moriya', 'lowme'),
    )
    ckpt_path = os.path.join(directory, 'emulator_6param_timeweighted_best.pt')
    
    bundle = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    cfg = bundle.get('config', {})
    ae_cfg = bundle.get('ae_config', {})
    
    input_dim = bundle['input_dim']
    latent_dim = bundle['latent_dim']
    flux_min = bundle['flux_min']
    flux_max = bundle['flux_max']
    z_mean = bundle['z_mean']
    z_std = bundle['z_std']
    time_grid = bundle['time_grid']
    wave_grid = bundle['wave_grid']
    n_time = bundle['n_time']
    n_wave = bundle['n_wave']
    
    # Load input normalization: prefer checkpoint, fallback to HDF5
    if 'X_mean' in bundle and 'X_std' in bundle:
        X_mean = bundle['X_mean']
        X_std = bundle['X_std']
    else:
        # Fallback to HDF5 for backward compatibility
        h5_path = '/data/zhangzy/retrain/svd_resnet/preprocessed_data_ae.h5'
        with h5py.File(h5_path, 'r') as f:
            X_mean = f['X_mean'][:input_dim]
            X_std = f['X_std'][:input_dim]
    
    # Build Emulator
    emulator = LowmeLatentEmulator(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dim=cfg.get('hidden_dim', 1024),
        num_blocks=cfg.get('num_blocks', 12),
    )
    emulator.load_state_dict(bundle['emulator_state_dict'])
    emulator.eval()
    
    # Build Decoder
    decoder = LowmeResNetDecoder(
        latent_dim=latent_dim,
        hidden_dim=ae_cfg.get('hidden_dim', 1024),
        output_dim=ae_cfg.get('input_dim', 10000),
        num_blocks=ae_cfg.get('num_blocks', 12),
    )
    decoder.load_state_dict(bundle['decoder_state_dict'])
    decoder.eval()
    
    return emulator, decoder, z_mean, z_std, flux_min, flux_max, time_grid, wave_grid, n_time, n_wave, X_mean, X_std


def clear_typeII_lowme_cache():
    """Clear LOWME model cache to free memory."""
    _load_typeII_lowme_torch_bundle.cache_clear()



def typeII_spectra_lowme(progenitor, ni_mass, log10_mdot, beta, rcsm, esn, **kwargs):
    """Type II spectra using the LOWME (Latent-space Optimized Waveform Emulator) surrogate.
    
    This model achieves ~0.95% RMSE on normalized flux (baseline ~5%).
    
    Parameters:
        progenitor: Initial ZAMS mass in solar masses (10-18)
        ni_mass: Mass of nickel-56 in solar masses (0.001-0.3)
        log10_mdot: log10 of mass-loss rate in M☉/yr
        beta: CSM density profile steepness (0.5-5)
        rcsm: CSM radius in 10^14 cm (1-10, will be multiplied by 1e14)
        esn: SN explosion energy in 10^51 erg (0.5-5)
        
    Returns:
        output namedtuple with fields:
        - spectrum: flux in erg/s/Hz (astropy quantity)
        - frequency: wavelength grid in Angstrom (astropy quantity)
        - time: time grid in days (astropy quantity)
    """
    rcsm_cm = rcsm * 1e14
    log10_mdot = np.abs(log10_mdot)
    
    device = kwargs.get('device', 'cpu')
    
    emulator, decoder, z_mean, z_std, flux_min, flux_max, time_grid, wave_grid, n_time, n_wave, X_mean, X_std = \
        _load_typeII_lowme_torch_bundle()
    
    emulator = emulator.to(device)
    decoder = decoder.to(device)
    z_mean = z_mean.to(device)
    z_std = z_std.to(device)
    
    # Prepare input: [progenitor, ni_mass, log10_mdot, beta, rcsm_cm, esn]
    ss = np.array([progenitor, ni_mass, log10_mdot, beta, rcsm_cm, esn]).T
    if isinstance(progenitor, float):
        ss = ss.reshape(1, -1)
    
    # Normalize input using HDF5 statistics
    ss = ((ss - X_mean) / X_std).astype(np.float32)
    
    with torch.no_grad():
        x_tensor = torch.from_numpy(ss).to(device)
        
        # Emulator predicts normalized latent code
        z_norm = emulator(x_tensor)
        
        # De-normalize latent code
        z = z_norm * z_std + z_mean
        
        # Decoder produces normalized spectrum [0, 1]
        pred_norm = decoder(z).cpu().numpy()
    
    # Convert to log10(flux)
    pred_logf = pred_norm * (flux_max - flux_min) + flux_min
    pred_logf = pred_logf.reshape(pred_logf.shape[0], n_time, n_wave)
    
    if isinstance(progenitor, float):
        pred_logf = pred_logf[0]
    
    # Convert to linear flux with units
    pred_flux = (10 ** pred_logf) * uu.erg / uu.s / uu.Hz
    
    output = namedtuple('output', ['spectrum', 'frequency', 'time'])
    return output(
        spectrum=pred_flux,
        frequency=np.array(wave_grid) * uu.Angstrom,
        time=np.array(time_grid) * uu.day,
    )


# Aliases for backward compatibility


# ========== Direct Regression Model ==========

class _DirectPhysicsEmbedding(nn.Module):
    """Physics parameter embedding for Direct Regression"""
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        feats = [x, x ** 2, torch.tanh(x), torch.log(torch.abs(x) + 1.1)]
        if x.shape[1] >= 2:
            feats.append(x[:, 0:1] * x[:, 1:2])  # M * Ni interaction
            feats.append(x[:, 1:2] * x[:, 5:6])  # Ni * Esn interaction
        return torch.cat(feats, dim=1)


class _DirectResBlock(nn.Module):
    """ResNet Block for Direct Regression"""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        h = torch.nn.functional.silu(self.norm1(x))
        h = self.fc1(h)
        h = self.dropout(h)
        h = torch.nn.functional.silu(self.norm2(h))
        h = self.fc2(h)
        return x + h


class DirectSpectrumRegressor(nn.Module):
    """Direct Regression: physical params -> spectrum (no latent space)"""
    def __init__(self, input_dim, output_dim, hidden_dim=2048, num_blocks=16):
        super().__init__()
        
        self.register_buffer('x_mean', torch.zeros(input_dim))
        self.register_buffer('x_std', torch.ones(input_dim))
        
        self.physics_embed = _DirectPhysicsEmbedding()
        embed_dim = input_dim * 4 + 2  # 4 features + 2 interactions
        
        self.input_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        
        self.blocks = nn.ModuleList([_DirectResBlock(hidden_dim, dropout=0.1) for _ in range(num_blocks)])
        
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x):
        x = (x - self.x_mean) / self.x_std
        x = self.physics_embed(x)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)


@lru_cache(maxsize=1)
def _load_typeII_direct_regression():
    """Load and cache the Direct Regression surrogate.
    
    Direct Regression uses a single-stage architecture:
    - Regressor: physical params -> spectrum (normalized [0,1]) directly
    """
    directory = os.environ.get('STELLA_DIRECT_DIR', '/data/zhangzy/retrain/redback_surrogates/surrogate_data/TypeII_Moriya/lowme')
    ckpt_path = os.path.join(directory, 'direct_regression_best.pt')
    
    bundle = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    cfg = bundle.get('config', {})
    input_dim = cfg['n_params']
    output_dim = cfg['output_dim']
    hidden_dim = cfg['hidden_dim']
    num_blocks = cfg['num_blocks']
    
    flux_min = bundle['flux_min']
    flux_max = bundle['flux_max']
    x_mean = bundle['x_mean']
    x_std = bundle['x_std']
    time_grid = bundle['time_grid']
    wave_grid = bundle['wave_grid']
    n_time = bundle['n_time']
    n_wave = bundle['n_wave']
    
    model = DirectSpectrumRegressor(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
    )
    model.load_state_dict(bundle['model_state_dict'])
    model.eval()
    
    return model, flux_min, flux_max, x_mean, x_std, time_grid, wave_grid, n_time, n_wave


def clear_typeII_direct_cache():
    """Clear Direct Regression model cache to free memory."""
    _load_typeII_direct_regression.cache_clear()


@citation_wrapper("https://ui.adsabs.harvard.edu/abs/2025arXiv250602107S/abstract, https://ui.adsabs.harvard.edu/abs/2023PASJ...75..634M/abstract")
def typeII_spectra_direct_regression(progenitor, ni_mass, log10_mdot, beta, rcsm, esn, **kwargs):
    """Type II spectra using the Direct Regression surrogate.
    
    This model directly predicts spectra from physical parameters without using
    a latent space representation, for comparison with two-stage approaches.
    
    Parameters:
        progenitor: Initial ZAMS mass in solar masses (10-18)
        ni_mass: Mass of nickel-56 in solar masses (0.001-0.3)
        log10_mdot: log10 of mass-loss rate in M☉/yr
        beta: CSM density profile steepness (0.5-5)
        rcsm: CSM radius in 10^14 cm (1-10, will be multiplied by 1e14)
        esn: SN explosion energy in 10^51 erg (0.5-5)
        
    Returns:
        output namedtuple with fields:
        - spectrum: flux in erg/s/Hz (astropy quantity)
        - frequency: wavelength grid in Angstrom (astropy quantity)
        - time: time grid in days (astropy quantity)
    """
    rcsm_cm = rcsm * 1e14
    log10_mdot = np.abs(log10_mdot)
    
    device = kwargs.get('device', 'cpu')
    
    model, flux_min, flux_max, x_mean, x_std, time_grid, wave_grid, n_time, n_wave = \
        _load_typeII_direct_regression()
    
    model = model.to(device)
    
    # Prepare input: [progenitor, ni_mass, log10_mdot, beta, rcsm_cm, esn]
    ss = np.array([progenitor, ni_mass, log10_mdot, beta, rcsm_cm, esn]).T
    if isinstance(progenitor, float):
        ss = ss.reshape(1, -1)
    
    # Normalize input
    ss = ((ss - x_mean.numpy()) / x_std.numpy()).astype(np.float32)
    
    with torch.no_grad():
        x_tensor = torch.from_numpy(ss).to(device)
        
        # Direct prediction of normalized spectrum [0, 1]
        pred_norm = model(x_tensor).cpu().numpy()
    
    # Convert to log10(flux)
    pred_logf = pred_norm * (flux_max - flux_min) + flux_min
    pred_logf = pred_logf.reshape(pred_logf.shape[0], n_time, n_wave)
    
    if isinstance(progenitor, float):
        pred_logf = pred_logf[0]
    
    # Convert to linear flux with units
    pred_flux = (10 ** pred_logf) * uu.erg / uu.s / uu.Hz
    
    output = namedtuple('output', ['spectrum', 'frequency', 'time'])
    return output(
        spectrum=pred_flux,
        frequency=np.array(wave_grid) * uu.Angstrom,
        time=np.array(time_grid) * uu.day,
    )


# Alias for convenience
typeII_spectra_direct = typeII_spectra_direct_regression


# ========== CleanModel CNN v3 (2D Convolutional AutoEncoder) ==========

class _CleanModelCNN2DEncoder(nn.Module):
    """CleanModel CNN Encoder: 100x100 -> latent_dim"""
    def __init__(self, latent_dim=256, base_ch=32, bottleneck_size=13):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, base_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_ch), nn.SiLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_ch*2), nn.SiLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_ch*2, base_ch*4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_ch*4), nn.SiLU())
        self.conv4 = nn.Sequential(
            nn.Conv2d(base_ch*4, base_ch*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch*4), nn.SiLU())
        self.fc = nn.Linear(base_ch * 4 * bottleneck_size * bottleneck_size, latent_dim)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.fc(x.view(x.size(0), -1))


class _CleanModelCNN2DDecoder(nn.Module):
    """CleanModel CNN Decoder: latent_dim -> 100x100"""
    def __init__(self, latent_dim=256, base_ch=32, bottleneck_size=13):
        super().__init__()
        self.bottleneck_size = bottleneck_size
        self.bottleneck_channels = base_ch * 4
        self.fc = nn.Linear(latent_dim, self.bottleneck_channels * bottleneck_size * bottleneck_size)
        self.up1 = nn.Sequential(
            nn.Conv2d(base_ch*4, base_ch*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch*4), nn.SiLU())
        self.up2 = nn.Sequential(
            nn.Conv2d(base_ch*4, base_ch*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch*2), nn.SiLU())
        self.up3 = nn.Sequential(
            nn.Conv2d(base_ch*2, base_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch), nn.SiLU())
        self.up4 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch), nn.SiLU())
        self.out_conv = nn.Conv2d(base_ch, 1, kernel_size=3, padding=1)
        
    def forward(self, z):
        x = self.fc(z).view(-1, self.bottleneck_channels, self.bottleneck_size, self.bottleneck_size)
        x = self.up1(x)
        x = nn.functional.interpolate(x, size=(25, 25), mode='bilinear', align_corners=False)
        x = self.up2(x)
        x = nn.functional.interpolate(x, size=(50, 50), mode='bilinear', align_corners=False)
        x = self.up3(x)
        x = nn.functional.interpolate(x, size=(100, 100), mode='bilinear', align_corners=False)
        x = self.up4(x)
        return self.out_conv(x)


class _CleanModelLatentEmulator(nn.Module):
    """CleanModel Latent Emulator: physical params -> latent code (no embedding)"""
    def __init__(self, input_dim, latent_dim, hidden_dim=512, num_blocks=8):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        blocks = []
        for _ in range(num_blocks):
            blocks.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.1),
            ))
        self.blocks = nn.ModuleList(blocks)
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        h = self.input_proj(x)
        for block in self.blocks:
            h = h + block(h)
        return self.output_proj(h)


@lru_cache(maxsize=1)
def _load_cleanmodel_cnn_v3():
    """Load and cache the CleanModel CNN v3 surrogate."""
    directory = os.path.join(data_folder, 'TypeII_Moriya', 'cleanmodel')
    ae_path = os.path.join(directory, 'ae_cnn_v3_best.pt')
    emulator_path = os.path.join(directory, 'emulator_cnn_v3_6param_best.pt')
    
    ae_ckpt = torch.load(ae_path, map_location='cpu', weights_only=False)
    em_ckpt = torch.load(emulator_path, map_location='cpu', weights_only=False)
    
    latent_dim = ae_ckpt['latent_dim']
    base_ch = ae_ckpt['base_channels']
    bottleneck_size = ae_ckpt['bottleneck_size']
    Y_min, Y_max = ae_ckpt['Y_min'], ae_ckpt['Y_max']
    wave_grid, time_grid = ae_ckpt['wave_grid'], ae_ckpt['time_grid']
    
    decoder = _CleanModelCNN2DDecoder(latent_dim, base_ch, bottleneck_size)
    # Extract decoder weights from full model
    decoder_state = {k.replace('decoder.', ''): v for k, v in ae_ckpt['model_state_dict'].items() if k.startswith('decoder.')}
    decoder.load_state_dict(decoder_state)
    decoder.eval()
    
    input_dim = em_ckpt['input_dim']
    hidden_dim = em_ckpt['hidden_dim']
    num_blocks = em_ckpt['num_blocks']
    z_mean, z_std = em_ckpt['z_mean'], em_ckpt['z_std']
    X_mean, X_std = em_ckpt['X_mean'], em_ckpt['X_std']
    
    emulator = _CleanModelLatentEmulator(input_dim, latent_dim, hidden_dim, num_blocks)
    emulator.load_state_dict(em_ckpt['emulator_state_dict'])
    emulator.eval()
    
    return emulator, decoder, z_mean, z_std, Y_min, Y_max, wave_grid, time_grid, X_mean, X_std


def clear_cleanmodel_cache():
    """Clear CleanModel cache to free memory."""
    _load_cleanmodel_cnn_v3.cache_clear()


@citation_wrapper("https://ui.adsabs.harvard.edu/abs/2025arXiv250602107S/abstract, https://ui.adsabs.harvard.edu/abs/2023PASJ...75..634M/abstract")
def typeII_spectra_cleanmodel(mass, ni_mass, mixing, energy, Menv, R0, **kwargs):
    """Type II spectra using the CleanModel CNN v3 surrogate.
    
    CleanModel uses a 2D Convolutional AutoEncoder architecture for improved
    spectral structure preservation with time-weighted loss.
    
    Parameters:
        mass: Progenitor ZAMS mass in solar masses (10-18)
        ni_mass: Mass of nickel-56 in solar masses (0.0001-0.3)
        mixing: Mixing code (0=cm, 1=fm, 2=hm) or continuous (0-2)
        energy: Explosion energy in 10^51 erg (0.2-5)
        Menv: Envelope mass in solar masses (7.2-9.5)
        R0: Initial radius in solar radii (510-970)
        
    Returns:
        output namedtuple with fields:
        - spectrum: flux in erg/s/Hz (astropy quantity)
        - frequency: wavelength grid in Angstrom (astropy quantity)
        - time: time grid in days (astropy quantity)
    """
    device = kwargs.get('device', 'cpu')
    
    emulator, decoder, z_mean, z_std, Y_min, Y_max, wave_grid, time_grid, X_mean, X_std = _load_cleanmodel_cnn_v3()
    
    emulator = emulator.to(device)
    decoder = decoder.to(device)
    z_mean = z_mean.to(device)
    z_std = z_std.to(device)
    
    # Input: [mass, ni_mass, mixing, energy, Menv, R0]
    ss = np.array([mass, ni_mass, mixing, energy, Menv, R0]).T
    if isinstance(mass, float):
        ss = ss.reshape(1, -1)
    ss = ((ss - X_mean) / X_std).astype(np.float32)
    
    with torch.no_grad():
        x_tensor = torch.from_numpy(ss).to(device)
        z_norm = emulator(x_tensor)
        z = z_norm * z_std + z_mean
        pred_norm = decoder(z).squeeze(1).cpu().numpy()  # Remove channel dim
    
    pred_logf = pred_norm * (Y_max - Y_min) + Y_min
    
    if isinstance(mass, float):
        pred_logf = pred_logf[0]
    
    pred_flux = (10 ** pred_logf) * uu.erg / uu.s / uu.Hz
    
    output = namedtuple('output', ['spectrum', 'frequency', 'time'])
    return output(
        spectrum=pred_flux,
        frequency=np.array(wave_grid) * uu.Angstrom,
        time=np.array(time_grid) * uu.day,
    )
