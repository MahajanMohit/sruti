package dev.sruti.ui

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
import androidx.compose.runtime.LaunchedEffect
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
import dev.sruti.llm.DeviceCapabilities
import dev.sruti.llm.NativeBackends
import dev.sruti.work.ModelJobState

private object Routes {
    const val CHAT = "chat"
    const val LIBRARY = "library"
    const val BROWSE = "browse"
    const val SETTINGS = "settings"
    const val BENCHMARK = "benchmark"
    const val ABOUT = "about"

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

                SettingsScreen(
                    tokenSet = library.tokenSet,
                    defaultQuant = library.defaultQuant,
                    onSetToken = viewModel::setToken,
                    onSetQuant = viewModel::setDefaultQuant,
                    keepCheckpoints = library.keepCheckpoints,
                    onSetKeepCheckpoints = viewModel::setKeepCheckpoints,
                    onRunBenchmark = { navController.navigate(Routes.BENCHMARK) },
                    onOpenAbout = { navController.navigate(Routes.ABOUT) },
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
