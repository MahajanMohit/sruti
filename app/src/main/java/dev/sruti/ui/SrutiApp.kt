package dev.sruti.ui

import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import dev.sruti.ui.library.BrowseScreen
import dev.sruti.ui.library.LibraryScreen
import dev.sruti.ui.library.LibraryViewModel
import dev.sruti.ui.library.SettingsScreen
import dev.sruti.ui.phase0.Phase0Screen
import dev.sruti.work.ModelJobState

private object Routes {
    const val LIBRARY = "library"
    const val BROWSE = "browse"
    const val SETTINGS = "settings"
    const val BENCHMARK = "benchmark"
}

@Composable
fun SrutiApp() {
    val navController = rememberNavController()
    val viewModel: LibraryViewModel = hiltViewModel()

    val library by viewModel.library.collectAsStateWithLifecycle()
    val browse by viewModel.browse.collectAsStateWithLifecycle()

    NavHost(navController = navController, startDestination = Routes.LIBRARY) {
        composable(Routes.LIBRARY) {
            LibraryScreen(
                state = library,
                onBrowse = { navController.navigate(Routes.BROWSE) },
                onDeleteModel = viewModel::delete,
                onDeleteCheckpoint = viewModel::delete,
                onCancelJob = viewModel::cancelJob,
                onDismissJob = viewModel::acknowledgeJob,
                onOpenSettings = { navController.navigate(Routes.SETTINGS) },
            )
        }

        composable(Routes.BROWSE) {
            BrowseScreen(
                state = browse,
                defaultQuant = library.defaultQuant,
                tokenSet = library.tokenSet,
                busy = library.job is ModelJobState.Running,
                onQueryChanged = viewModel::onQueryChanged,
                onInspect = viewModel::inspect,
                onAcquire = { repoId ->
                    viewModel.startAcquire(repoId)
                    // Back to the library, where the job card shows progress.
                    navController.popBackStack()
                },
                onBack = { navController.popBackStack() },
                onOpenSettings = { navController.navigate(Routes.SETTINGS) },
            )
        }

        composable(Routes.SETTINGS) {
            SettingsScreen(
                tokenSet = library.tokenSet,
                defaultQuant = library.defaultQuant,
                onSetToken = viewModel::setToken,
                onSetQuant = viewModel::setDefaultQuant,
                onRunBenchmark = { navController.navigate(Routes.BENCHMARK) },
                onBack = { navController.popBackStack() },
            )
        }

        composable(Routes.BENCHMARK) {
            Phase0Screen()
        }
    }
}
