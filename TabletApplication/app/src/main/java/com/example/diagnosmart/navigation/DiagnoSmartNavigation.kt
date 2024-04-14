package com.example.diagnosmart.navigation

import androidx.navigation.NavHostController
import com.example.diagnosmart.navigation.NavigationScreens.HOME_SCREEN
import com.example.diagnosmart.navigation.NavigationScreens.LOGIN_SCREEN
import com.example.diagnosmart.navigation.NavigationScreens.RESULTS_SCREEN
import com.example.diagnosmart.navigation.NavigationScreens.SEARCH_SCREEN
import com.example.diagnosmart.navigation.NavigationScreens.SETTINGS_SCREEN
import com.example.diagnosmart.navigation.NavigationScreens.SPLASH_SCREEN

/**
 * Screens used in [DiagnoSmartDestinations]
 */
private object NavigationScreens {

    const val SPLASH_SCREEN = "splash"
    const val LOGIN_SCREEN = "login"
    const val HOME_SCREEN = "home"
    const val RESULTS_SCREEN = "results"
    const val SEARCH_SCREEN = "search"
    const val MAIN_SCREEN = "main"
    const val SETTINGS_SCREEN = "settings"
}

///**
// * Arguments used in [InspectorDestinations] routes
// */
//object InspectorDestinationsArgs {
//    const val FIXATION_ID = "fixationId"
//    const val FORCE_PRINT = "forcePrint"
//    const val USER_MESSAGE_ARG = "userMessage"
//    const val PHOTO_URI = "photoUri"
//}

/**
 * Destinations used in the [MainActivity]
 */
object DiagnoSmartDestinations {
//    const val AUTHORIZATION_ROUTE = "$LOGIN_SCREEN?$USER_MESSAGE_ARG={$USER_MESSAGE_ARG}"

    const val SPLASH_ROUTE = SPLASH_SCREEN
    const val LOGIN_ROUTE = LOGIN_SCREEN
    const val HOME_ROUTE = HOME_SCREEN
    const val RESULTS_ROUTE = RESULTS_SCREEN
    const val SEARCH_ROUTE = SEARCH_SCREEN
    const val SETTINGS_ROUTE = SETTINGS_SCREEN
}

/**
 * Models the navigation actions in the app.
 */
class DiagnoSmartNavigationActions(private val navController: NavHostController) {

    fun navigateToSplash() {
        navController.navigate(SPLASH_SCREEN)
    }

    fun navigateToSearch() {
        navController.navigate(SEARCH_SCREEN)
    }

    fun navigateToLogin() {
        navController.navigate(LOGIN_SCREEN) {
            popUpTo(LOGIN_SCREEN) {
                inclusive = true
            }
        }
    }

    fun navigateToHome() {
        navController.navigate(HOME_SCREEN)
    }

    fun navigateToResults() {
        navController.navigate(RESULTS_SCREEN)
    }

    fun navigateToSetting() {
        navController.navigate(SETTINGS_SCREEN)
    }

//    fun navigateToReviewPhotoTicket(fixationId: Long) {
//        navController.navigate("$PRINTER_REVIEW_PHOTO_SCREEN/$fixationId")
//    }
}
