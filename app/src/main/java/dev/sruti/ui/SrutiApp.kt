package dev.sruti.ui

import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInHorizontally
import androidx.compose.animation.slideOutHorizontally
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.outlined.Chat
import androidx.compose.material.icons.outlined.Layers
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.navigation.NavDestination.Companion.hierarchy
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.NavType
import androidx.navigation.navArgument
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import dev.sruti.ui.chat.ChatScreen
import dev.sruti.ui.chat.ChatViewModel
import dev.sruti.ui.library.BrowseScreen
import dev.sruti.ui.library.LibraryScreen
import dev.sruti.ui.library.LibraryViewModel
import dev.sruti.ui.library.AboutScreen
import dev.sruti.ui.library.ModelDetailScreen
import dev.sruti.ui.library.SettingsScreen
import dev.sruti.ui.phase0.Phase0Screen
import dev.sruti.ui.skills.SkillEditorScreen
import dev.sruti.ui.skills.SkillsScreen
import dev.sruti.ui.skills.SkillsViewModel
import dev.sruti.llm.DeviceCapabilities
import dev.sruti.llm.NativeBackends
import dev.sruti.ui.theme.Haptic
import dev.sruti.ui.theme.LocalHaptics
import dev.sruti.ui.theme.Motion
import dev.sruti.work.ModelJobState

private object Routes {
    const val CHAT = "chat"
    const val LIBRARY = "library"
    const val BROWSE = "browse"
    const val SETTINGS = "settings"
    const val BENCHMARK = "benchmark"
    const val ABOUT = "about"
    const val SKILLS = "skills"
    const val SKILL_EDITOR = "skill_editor"

    /** The model's absolute path, encoded — it contains slashes. */
    const val MODEL_DETAIL = "model/{path}"

    fun modelDetail(path: String): String =
        "model/" + android.net.Uri.encode(path)
}

private data class TopLevelDestination(
    val route: String,
    val label: String,
    val icon: @Composable () -> Unit,
)

@Composable
fun SrutiApp() {
    val navController = rememberNavController()
    val context = LocalContext.current
    val haptics = LocalHaptics.current

    val topLevel = listOf(
        TopLevelDestination(Routes.CHAT, "Chat") {
            Icon(Icons.AutoMirrored.Outlined.Chat, contentDescription = null)
        },
        TopLevelDestination(Routes.LIBRARY, "Models") {
            Icon(Icons.Outlined.Layers, contentDescription = null)
        },
    )

    val backStackEntry by navController.currentBackStackEntryAsState()
    val currentDestination = backStackEntry?.destination
    val showBottomBar = topLevel.any { it.route == currentDestination?.route }

    Scaffold(
        bottomBar = {
            if (showBottomBar) {
                NavigationBar {
                    topLevel.forEach { destination ->
                        val selected = currentDestination?.hierarchy
                            ?.any { it.route == destination.route } == true
                        NavigationBarItem(
                            selected = selected,
                            onClick = {
                                if (!selected) haptics.play(Haptic.Select)
                                navController.navigate(destination.route) {
                                    // Switching tabs should not stack duplicates,
                                    // and returning to a tab should restore where
                                    // the user left it.
                                    popUpTo(navController.graph.findStartDestination().id) {
                                        saveState = true
                                    }
                                    launchSingleTop = true
                                    restoreState = true
                                }
                            },
                            icon = destination.icon,
                            label = { Text(destination.label) },
                        )
                    }
                }
            }
        },
    ) { insets ->
        NavHost(
            navController = navController,
            startDestination = Routes.CHAT,
            modifier = Modifier.padding(bottom = insets.calculateBottomPadding()),

            // Forward motion is a short slide plus a fade; the screen being left
            // moves a fraction of the distance rather than sliding fully out, so
            // the two feel connected rather than like separate cards on a stack.
            //
            // Springs, not durations, because a predictive-back gesture drives
            // this animation from the user's finger — it has to be interruptible
            // and reversible mid-flight, which a fixed-duration curve is not.
            enterTransition = {
                slideInHorizontally(Motion.expressive()) { it / 5 } + fadeIn(Motion.standard())
            },
            exitTransition = {
                slideOutHorizontally(Motion.expressive()) { -it / 8 } + fadeOut(Motion.quick())
            },
            popEnterTransition = {
                slideInHorizontally(Motion.expressive()) { -it / 8 } + fadeIn(Motion.standard())
            },
            popExitTransition = {
                slideOutHorizontally(Motion.expressive()) { it / 5 } + fadeOut(Motion.quick())
            },
        ) {
            composable(Routes.CHAT) {
                val viewModel: ChatViewModel = hiltViewModel()
                val state by viewModel.uiState.collectAsStateWithLifecycle()

                ChatScreen(
                    state = state,
                    onSend = viewModel::send,
                    onStop = viewModel::stop,
                    onNewConversation = viewModel::startNewConversation,
                    onOpenConversation = viewModel::openConversation,
                    onDeleteConversation = viewModel::deleteConversation,
                    onSelectModel = viewModel::selectModel,
                    onOpenModels = { navController.navigate(Routes.LIBRARY) },
                    onSetAgentMode = viewModel::setAgentMode,
                    onResolveConfirmation = viewModel::resolveConfirmation,
                    onOpenSettings = { navController.navigate(Routes.SETTINGS) },
                )
            }

            composable(Routes.LIBRARY) {
                val viewModel: LibraryViewModel = hiltViewModel()
                val library by viewModel.library.collectAsStateWithLifecycle()

                LibraryScreen(
                    state = library,
                    onBrowse = { navController.navigate(Routes.BROWSE) },
                    onDeleteModel = viewModel::delete,
                    onOpenModel = { model ->
                        navController.navigate(Routes.modelDetail(model.file.absolutePath))
                    },
                    onDeleteCheckpoint = viewModel::delete,
                    onCancelJob = viewModel::cancelJob,
                    onDismissJob = viewModel::acknowledgeJob,
                    onOpenSettings = { navController.navigate(Routes.SETTINGS) },
                )
            }

            composable(Routes.BROWSE) {
                val viewModel: LibraryViewModel = hiltViewModel()
                val library by viewModel.library.collectAsStateWithLifecycle()
                val browse by viewModel.browse.collectAsStateWithLifecycle()

                BrowseScreen(
                    state = browse,
                    defaultQuant = library.defaultQuant,
                    tokenSet = library.tokenSet,
                    busy = library.job is ModelJobState.Running,
                    onQueryChanged = viewModel::onQueryChanged,
                    onFiltersChanged = viewModel::onFiltersChanged,
                    onInspect = viewModel::inspect,
                    onAcquire = { repoId ->
                        viewModel.startAcquire(repoId)
                        navController.popBackStack()
                    },
                    onOpenModelPage = { repoId -> openUrl(context, viewModel.modelPageUrl(repoId)) },
                    onBack = { navController.popBackStack() },
                    onOpenSettings = { navController.navigate(Routes.SETTINGS) },
                )
            }

            composable(Routes.SETTINGS) {
                val viewModel: LibraryViewModel = hiltViewModel()
                val library by viewModel.library.collectAsStateWithLifecycle()

                // Termux may have been installed or permitted while the user was
                // away from the app; the answer is only correct when re-read.
                LaunchedEffect(Unit) { viewModel.refreshTermuxStatus() }

                SettingsScreen(
                    tokenSet = library.tokenSet,
                    defaultQuant = library.defaultQuant,
                    onSetToken = viewModel::setToken,
                    onSetQuant = viewModel::setDefaultQuant,
                    keepCheckpoints = library.keepCheckpoints,
                    onSetKeepCheckpoints = viewModel::setKeepCheckpoints,
                    shellEnabled = library.shellEnabled,
                    termuxInstalled = library.termuxInstalled,
                    termuxPermitted = library.termuxPermitted,
                    onSetShellEnabled = viewModel::setShellEnabled,
                    onRequestTermuxPermission = { requestTermuxPermission(context) },
                    versionName = dev.sruti.BuildConfig.VERSION_NAME,
                    updateStatus = library.updateStatus,
                    checkingUpdate = library.checkingUpdate,
                    onCheckForUpdate = viewModel::checkForUpdate,
                    onOpenUrl = { url -> openUrl(context, url) },
                    onRunBenchmark = { navController.navigate(Routes.BENCHMARK) },
                    onOpenAbout = { navController.navigate(Routes.ABOUT) },
                    onOpenSkills = { navController.navigate(Routes.SKILLS) },
                    onBack = { navController.popBackStack() },
                )
            }

            composable(
                Routes.MODEL_DETAIL,
                arguments = listOf(navArgument("path") { type = NavType.StringType }),
            ) { entry ->
                val viewModel: LibraryViewModel = hiltViewModel()
                val detail by viewModel.detail.collectAsStateWithLifecycle()
                val path = entry.arguments?.getString("path").orEmpty()

                // Keyed on the path so navigating from one model to another
                // re-reads rather than showing the previous model's header.
                LaunchedEffect(path) { viewModel.openDetail(path) }

                detail.model?.let { model ->
                    ModelDetailScreen(
                        model = model,
                        facts = detail.facts,
                        loading = detail.loading,
                        onBack = { navController.popBackStack() },
                    )
                }
            }

            composable(Routes.SKILLS) {
                val viewModel: SkillsViewModel = hiltViewModel()
                val skills by viewModel.state.collectAsStateWithLifecycle()

                // The picker is registered here rather than in the screen so the
                // screen stays a pure function of its state.
                val picker = rememberLauncherForActivityResult(
                    ActivityResultContracts.OpenDocument(),
                ) { uri -> uri?.let(viewModel::importFrom) }

                SkillsScreen(
                    state = skills,
                    missingTools = viewModel::missingTools,
                    onNew = {
                        viewModel.startNew()
                        navController.navigate(Routes.SKILL_EDITOR)
                    },
                    onEdit = { skill ->
                        viewModel.startEditing(skill)
                        navController.navigate(Routes.SKILL_EDITOR)
                    },
                    onDelete = viewModel::delete,
                    // Skills are markdown, but a file manager will not always
                    // label them as such, so plain text is accepted too.
                    onImportFile = { picker.launch(arrayOf("text/markdown", "text/plain", "*/*")) },
                    onImportUrl = viewModel::importFrom,
                    onDismissError = viewModel::clearError,
                    onBack = { navController.popBackStack() },
                )
            }

            composable(Routes.SKILL_EDITOR) { entry ->
                // Scoped to the skills route so the editor shares its view model
                // and the text being edited survives the navigation.
                val parent = remember(entry) { navController.getBackStackEntry(Routes.SKILLS) }
                val viewModel: SkillsViewModel = hiltViewModel(parent)
                val editor by viewModel.editor.collectAsStateWithLifecycle()

                SkillEditorScreen(
                    state = editor,
                    onTextChanged = viewModel::onEditorTextChanged,
                    onSave = viewModel::save,
                    onBack = { navController.popBackStack() },
                )
            }

            composable(Routes.BENCHMARK) {
                Phase0Screen()
            }

            composable(Routes.ABOUT) {
                AboutScreen(
                    backendCount = NativeBackends.backendCount(),
                    backendDetail = NativeBackends.diagnostics(),
                    threadCount = DeviceCapabilities.recommendedThreadCount(),
                    onBack = { navController.popBackStack() },
                )
            }
        }
    }
}

/**
 * Opens a web page.
 *
 * Used for accepting a gated model's licence, which can only be done on the
 * Hugging Face site — there is no API for it.
 */
private fun openUrl(context: android.content.Context, url: String) {
    runCatching {
        context.startActivity(
            android.content.Intent(android.content.Intent.ACTION_VIEW, android.net.Uri.parse(url))
                .addFlags(android.content.Intent.FLAG_ACTIVITY_NEW_TASK),
        )
    }
}

/**
 * Asks Termux for the RUN_COMMAND permission.
 *
 * A normal runtime request: Termux declares it as a dangerous permission, so the
 * system prompt is the only way to obtain it. Nothing here can grant it, and a
 * user who declines simply keeps a working app without the shell tier.
 */
private fun requestTermuxPermission(context: android.content.Context) {
    val activity = generateSequence(context) { (it as? android.content.ContextWrapper)?.baseContext }
        .filterIsInstance<android.app.Activity>()
        .firstOrNull() ?: return

    androidx.core.app.ActivityCompat.requestPermissions(
        activity,
        arrayOf("com.termux.permission.RUN_COMMAND"),
        /*requestCode=*/1001,
    )
}
