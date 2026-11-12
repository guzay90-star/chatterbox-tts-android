package com.acul3.chatterboxtts.ui

import android.content.Context
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import com.acul3.chatterboxtts.tts.VoiceStyles
import com.acul3.chatterboxtts.tts.VoicePack
import com.acul3.chatterboxtts.tts.VoiceStyle
import android.media.MediaScannerConnection
import android.widget.Toast
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.acul3.chatterboxtts.tts.TTSViewModel
import com.acul3.chatterboxtts.tts.TTSState
import kotlinx.coroutines.launch
import java.io.File

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen(viewModel: TTSViewModel = viewModel()) {
    val context = LocalContext.current
    val state by viewModel.state.collectAsState()
    val scope = rememberCoroutineScope()
    // VOICE_V2_PATCH — import real speaker conditioning as a ZIP voice pack.
    val voiceImport = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null) viewModel.importVoicePack(context, uri)
    }
    val wavExport = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("audio/wav")) { uri ->
        if (uri != null) {
            try {
                viewModel.saveWavToUri(context, uri)
                Toast.makeText(context, "WAV сохранён", Toast.LENGTH_SHORT).show()
            } catch (e: Exception) {
                Toast.makeText(context, "Ошибка сохранения: ${e.message}", Toast.LENGTH_LONG).show()
            }
        }
    }
    LaunchedEffect(Unit) { viewModel.refreshVoicePacks(context) }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text("Chatterbox TTS", fontWeight = FontWeight.Bold)
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer,
                    titleContentColor = MaterialTheme.colorScheme.onPrimaryContainer,
                )
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp)
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Text input
            OutlinedTextField(
                value = state.inputText,
                onValueChange = { viewModel.updateText(it) },
                label = { Text("Enter text to speak") },
                modifier = Modifier
                    .fillMaxWidth()
                    .heightIn(min = 120.dp),
                maxLines = 8,
                enabled = !state.isGenerating
            )

            // Language selector
            LanguageDropdown(
                selectedLanguage = state.selectedLanguage,
                onLanguageSelected = { viewModel.updateLanguage(it) },
                enabled = !state.isGenerating
            )

            // Speaker selection uses neural conditioning from voice packs.
            VoicePackDropdown(
                packs = state.voicePacks,
                selectedId = state.selectedVoicePack,
                enabled = !state.isGenerating,
                onSelect = viewModel::setVoicePack
            )
            OutlinedButton(
                onClick = { voiceImport.launch("application/zip") },
                enabled = !state.isGenerating,
                modifier = Modifier.fillMaxWidth()
            ) { Text("Импортировать голосовой пакет ZIP") }
            Text(
                "ZIP должен содержать параметры подготовленного голоса. Обычный WAV пока нельзя использовать для клонирования.",
                style = MaterialTheme.typography.bodySmall
            )

            VoiceStyleDropdown(
                selectedId = state.styleId,
                enabled = !state.isGenerating,
                onSelect = viewModel::setStyle
            )
            Text("Стили изменяют звучание выбранного голоса, но не создают нового диктора.", style = MaterialTheme.typography.bodySmall)
            Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Text("Скорость: ${"%.2f".format(java.util.Locale.US, state.speed)}×")
                Slider(
                    value = state.speed,
                    onValueChange = viewModel::setSpeed,
                    valueRange = 0.65f..1.5f,
                    enabled = !state.isGenerating
                )
                Text("Высота: ${"%.1f".format(java.util.Locale.US, state.pitch)} полутона")
                Slider(
                    value = state.pitch,
                    onValueChange = viewModel::setPitch,
                    valueRange = -7f..7f,
                    enabled = !state.isGenerating
                )
                Text("Выразительность: ${"%.2f".format(java.util.Locale.US, state.expression)}")
                Slider(
                    value = state.expression,
                    onValueChange = viewModel::setExpression,
                    valueRange = 0.25f..1.5f,
                    enabled = !state.isGenerating
                )
            }

            // Generate button
            Button(
                onClick = { scope.launch { viewModel.generate(context) } },
                modifier = Modifier.fillMaxWidth(),
                enabled = state.inputText.isNotBlank() && !state.isGenerating
            ) {
                if (state.isGenerating) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(20.dp),
                        strokeWidth = 2.dp,
                        color = MaterialTheme.colorScheme.onPrimary
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                }
                Text(if (state.isGenerating) "Generating..." else "Generate Speech")
            }

            // Progress section
            if (state.isGenerating || state.progressMessage.isNotEmpty()) {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.surfaceVariant
                    )
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        if (state.isGenerating) {
                            LinearProgressIndicator(
                                progress = { state.progress },
                                modifier = Modifier.fillMaxWidth(),
                            )
                            Spacer(modifier = Modifier.height(8.dp))
                        }
                        Text(
                            text = state.progressMessage,
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
            }

            // Error display
            if (state.error != null) {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.errorContainer
                    )
                ) {
                    Text(
                        text = state.error!!,
                        modifier = Modifier.padding(16.dp),
                        color = MaterialTheme.colorScheme.onErrorContainer,
                        style = MaterialTheme.typography.bodyMedium
                    )
                }
            }

            // Playback controls
            if (state.audioReady) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Button(
                        onClick = {
                            if (state.isPlaying) viewModel.stopPlayback()
                            else viewModel.playAudio(context)
                        },
                        modifier = Modifier.weight(1f)
                    ) {
                        Icon(
                            Icons.Default.PlayArrow,
                            contentDescription = null,
                            modifier = Modifier.size(18.dp)
                        )
                        Spacer(modifier = Modifier.width(4.dp))
                        Text(if (state.isPlaying) "Stop" else "Play")
                    }

                    OutlinedButton(
                        onClick = {
                            wavExport.launch("chatterbox_${System.currentTimeMillis()}.wav")
                        },
                        modifier = Modifier.weight(1f)
                    ) {
                        Text("Save WAV")
                    }
                }
            }

            // Model status
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f)
                )
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(
                        "Model Status",
                        style = MaterialTheme.typography.labelLarge,
                        fontWeight = FontWeight.Bold
                    )
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        text = state.modelStatus,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun LanguageDropdown(
    selectedLanguage: String,
    onLanguageSelected: (String) -> Unit,
    enabled: Boolean = true
) {
    val languages = listOf(
        "en" to "English",
        "fr" to "French",
        "de" to "German",
        "es" to "Spanish",
        "pt" to "Portuguese",
        "it" to "Italian",
        "nl" to "Dutch",
        "pl" to "Polish",
        "cs" to "Czech",
        "ro" to "Romanian",
        "hu" to "Hungarian",
        "fi" to "Finnish",
        "da" to "Danish",
        "sv" to "Swedish",
        "tr" to "Turkish",
        "ru" to "Russian",
        "zh" to "Chinese",
        "ja" to "Japanese",
        "ko" to "Korean",
        "ms" to "Malay",
        "id" to "Indonesian",
        "th" to "Thai",
        "vi" to "Vietnamese"
    )

    var expanded by remember { mutableStateOf(false) }
    val selectedLabel = languages.find { it.first == selectedLanguage }?.second ?: selectedLanguage

    ExposedDropdownMenuBox(
        expanded = expanded,
        onExpandedChange = { if (enabled) expanded = !expanded }
    ) {
        OutlinedTextField(
            value = "[$selectedLanguage] $selectedLabel",
            onValueChange = {},
            readOnly = true,
            label = { Text("Language") },
            trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded) },
            modifier = Modifier
                .fillMaxWidth()
                .menuAnchor(),
            enabled = enabled
        )
        ExposedDropdownMenu(
            expanded = expanded,
            onDismissRequest = { expanded = false }
        ) {
            languages.forEach { (code, name) ->
                DropdownMenuItem(
                    text = { Text("[$code] $name") },
                    onClick = {
                        onLanguageSelected(code)
                        expanded = false
                    }
                )
            }
        }
    }
}

private fun saveWav(context: Context, viewModel: TTSViewModel) {
    val downloadsDir = android.os.Environment.getExternalStoragePublicDirectory(
        android.os.Environment.DIRECTORY_DOWNLOADS
    )
    val file = File(downloadsDir, "chatterbox_${System.currentTimeMillis()}.wav")
    try {
        viewModel.saveWavToFile(file)
        MediaScannerConnection.scanFile(context, arrayOf(file.absolutePath), null, null)
        Toast.makeText(context, "Saved to ${file.name}", Toast.LENGTH_SHORT).show()
    } catch (e: Exception) {
        Toast.makeText(context, "Save failed: ${e.message}", Toast.LENGTH_SHORT).show()
    }
}


@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun VoicePackDropdown(
    packs: List<VoicePack>, selectedId: String, enabled: Boolean,
    onSelect: (String) -> Unit
) {
    var expanded by remember { mutableStateOf(false) }
    ExposedDropdownMenuBox(expanded = expanded, onExpandedChange = { if (enabled) expanded = !expanded }) {
        OutlinedTextField(
            value = packs.firstOrNull { it.id == selectedId }?.title ?: "Встроенный голос",
            onValueChange = {},
            label = { Text("Голос / нейронный профиль") },
            trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded) },
            readOnly = true,
            enabled = enabled,
            modifier = Modifier.fillMaxWidth().menuAnchor()
        )
        ExposedDropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
            packs.forEach { pack ->
                DropdownMenuItem(text = { Text(pack.title) }, onClick = { onSelect(pack.id); expanded = false })
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun VoiceStyleDropdown(selectedId: String, enabled: Boolean, onSelect: (String) -> Unit) {
    var expanded by remember { mutableStateOf(false) }
    val style = VoiceStyles.get(selectedId)
    ExposedDropdownMenuBox(expanded = expanded, onExpandedChange = { if (enabled) expanded = !expanded }) {
        OutlinedTextField(
            value = style.name,
            onValueChange = {},
            label = { Text("Стиль звучания") },
            supportingText = { Text(style.description) },
            trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded) },
            readOnly = true,
            enabled = enabled,
            modifier = Modifier.fillMaxWidth().menuAnchor()
        )
        ExposedDropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
            VoiceStyles.all.forEach { item ->
                DropdownMenuItem(text = { Text("${item.name} — ${item.description}") }, onClick = {
                    onSelect(item.id)
                    expanded = false
                })
            }
        }
    }
}
