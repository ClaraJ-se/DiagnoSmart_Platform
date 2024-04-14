package com.example.diagnosmart.navigation

import android.app.Activity
import android.util.Log
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.wrapContentSize
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.DrawerState
import androidx.compose.material3.DrawerValue
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Text
import androidx.compose.material3.rememberDrawerState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.res.colorResource
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.LifecycleOwner
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.example.diagnosmart.home.HomeScreen
import com.example.diagnosmart.login.LoginScreen
import com.example.diagnosmart.main.DiagnoSmartViewModel
import com.example.symtomsavvy.R
import kotlinx.coroutines.CoroutineScope

@Composable
fun DiagnoSmartNavGraph(
        modifier: Modifier = Modifier,
        navController: NavHostController = rememberNavController(),
        coroutineScope: CoroutineScope = rememberCoroutineScope(),
        drawerState: DrawerState = rememberDrawerState(initialValue = DrawerValue.Closed),
        startDestination: String = DiagnoSmartDestinations.LOGIN_ROUTE,
        navActions: DiagnoSmartNavigationActions = remember(navController) {
            DiagnoSmartNavigationActions(navController)
        },
        diagnoSmartViewModel: DiagnoSmartViewModel
) {
    val currentNavBackStackEntry by navController.currentBackStackEntryAsState()
    val currentRoute = currentNavBackStackEntry?.destination?.route ?: startDestination
    val mainState by diagnoSmartViewModel.mainState.collectAsState()
    val context = LocalContext.current
    val activity = (context as Activity)
    val uiState by diagnoSmartViewModel.mainState.collectAsState()
    var searchText by remember { mutableStateOf(TextFieldValue()) }

    ComposableLifecycle { source, event ->
        when (event) {
            Lifecycle.Event.ON_CREATE -> {
                Log.d("TAG", "MainScreen: onCreate")
            }

            Lifecycle.Event.ON_START -> {
                Log.d("TAG", "MainScreen: ON_START")
            }

            Lifecycle.Event.ON_RESUME -> {
            }

            Lifecycle.Event.ON_PAUSE -> {
                Log.d("TAG", "MainScreen: ON_PAUSE")
            }

            Lifecycle.Event.ON_STOP -> {
                Log.d("TAG", "MainScreen: ON_STOP")
            }

            Lifecycle.Event.ON_DESTROY -> {
                Log.d("TAG", "MainScreen: ON_DESTROY")
            }

            else -> {}
        }
    }


    Box(modifier = Modifier.fillMaxSize()) {

        Row(modifier = Modifier.fillMaxSize()) {

            if (uiState.isSessionActive) {
                Column(
                        modifier = Modifier
                            .width(120.dp)
                            .background(colorResource(id = R.color.color_left_navigation))
                            .fillMaxHeight(), verticalArrangement = Arrangement.Top,
                        horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    // Icon on top
                    TabImageButton(
                            Modifier
                                .width(100.dp)
                                .height(120.dp),
                            stringResource(id = R.string.tab_home), false,
                            R.drawable.home_ic, R.drawable.home_ic_unselected,
                            hidden = false,
                            onClick = {
                            })


                    Spacer(
                            modifier = Modifier
                                .height(10.dp)
                                .weight(1f)
                    )
                    // Three icons at the bottom

                    TabImageButton(
                            Modifier
                                .fillMaxSize()
                                .weight(0.5f),
                            stringResource(id = R.string.tab_1), false,
                            R.drawable.tab_1_selected, R.drawable.tab_1_unselected,
                            hidden = currentRoute != DiagnoSmartDestinations.HOME_ROUTE,
                            onClick = {
                            })
                    Spacer(modifier = Modifier.width(10.dp))
                    TabImageButton(
                            Modifier
                                .fillMaxSize()
                                .weight(0.5f),
                            stringResource(id = R.string.tab_2), false,
                            R.drawable.tab_2_selected, R.drawable.tab_2_unselected,
                            hidden = currentRoute != DiagnoSmartDestinations.HOME_ROUTE,
                            onClick = {
                            })
                    Spacer(modifier = Modifier.width(10.dp))
                    TabImageButton(
                            Modifier
                                .fillMaxSize()
                                .weight(0.5f),
                            stringResource(id = R.string.tab_3), false,
                            R.drawable.tab_3_selected, R.drawable.tab_3_unselected,
                            hidden = currentRoute != DiagnoSmartDestinations.HOME_ROUTE,
                            onClick = {
                            })
                }
            }

            Column(
                    modifier = Modifier
                        .weight(1f)
                        .background(colorResource(id = R.color.color_background)),
                    verticalArrangement = Arrangement.Top,
                    horizontalAlignment = Alignment.CenterHorizontally

            ) {

                if (uiState.isSessionActive) {
                    Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(50.dp)
                                .padding(8.dp),
                            verticalAlignment = Alignment.CenterVertically
                    ) {
                        SearchBar(
                                modifier = Modifier.weight(0.5f), searchText = searchText,
                                onSearchTextChanged = { searchText = it })


                        ButtonWithTextAndIcon(
                                Modifier, "Sort", R.drawable.sort_ic, 30, onClick = {})

                        Spacer(modifier = Modifier.weight(0.3f))


                        Text(
                                text = "TEST USER",
                                modifier = Modifier,
                                color = Color.Black,
                                textAlign = TextAlign.Center
                        )

                        IconButtonWithIcon(modifier = Modifier, 30, R.drawable.profile_small_ic,
                                onClick = {})

                        IconButtonWithIcon(modifier = Modifier, 30, R.drawable.menu_dots,
                                onClick = {})

                    }
                }

                diagnoSmartViewModel.checkIfSessionActive()
                setupNav(navController, startDestination, modifier, navActions)

                LaunchedEffect(uiState) {
                    when (uiState.isSessionActive) {
                        true -> navController.navigate(DiagnoSmartDestinations.HOME_ROUTE)
                        false -> navController.navigate(DiagnoSmartDestinations.LOGIN_ROUTE)
                    }
                }
            }

        }

    }
}

@Composable
fun SearchBar(
        modifier: Modifier = Modifier,
        searchText: TextFieldValue,
        onSearchTextChanged: (TextFieldValue) -> Unit
) {
    Row(
            modifier = modifier
                .fillMaxWidth()
                .height(56.dp)
                .padding(horizontal = 16.dp)
                .background(color = Color.White, shape = RoundedCornerShape(8.dp)),
            verticalAlignment = Alignment.CenterVertically
    ) {
        BasicTextField(
                value = searchText,
                onValueChange = { onSearchTextChanged(it) },
                singleLine = true,
                modifier = Modifier
                    .weight(1f)
                    .padding(end = 8.dp)

                    .padding(horizontal = 48.dp, vertical = 8.dp),
                textStyle = TextStyle(color = Color.Black),
                cursorBrush = SolidColor(Color.Black),
                decorationBox = { innerTextField ->
                    Box(contentAlignment = Alignment.CenterStart) {
                        if (searchText.text.isEmpty()) {
                            Text(
                                    text = "Search Patient",
                                    color = Color.Gray
                            )
                        }
                        innerTextField()
                    }
                }
        )
        Icon(
                modifier = Modifier.padding(end = 8.dp),
                imageVector = Icons.Default.Search,
                contentDescription = "Search",
                tint = Color.Black,

                )
    }
}

@Composable
fun setupNav(navController: NavHostController, startDestination: String, modifier: Modifier,
             navActions: DiagnoSmartNavigationActions) {

    NavHost(
            navController = navController,
            startDestination = startDestination,
            modifier = modifier
    ) {

        composable(DiagnoSmartDestinations.LOGIN_ROUTE) {
            LoginScreen(
                    onBack = { navController.popBackStack() },
                    onLoginComplete = {

                        navActions.navigateToHome()
                    },
                    openAlternativeLogin = {},
                    onSettingOpen = { navActions.navigateToSetting() },
                    onLoginError = {}
            )
        }

        composable(DiagnoSmartDestinations.HOME_ROUTE) {
            HomeScreen(
                    onBack = { navController.popBackStack() },
                    onAddPatient = { navActions.navigateToHome() },
                    onSettingOpen = { navActions.navigateToSetting() },
                    onLoginError = {}
            )
        }

        composable(DiagnoSmartDestinations.RESULTS_ROUTE) {
            LoginScreen(
                    onBack = { navController.popBackStack() },
                    onLoginComplete = { navActions.navigateToResults() },
                    openAlternativeLogin = {},
                    onSettingOpen = { navActions.navigateToHome() },
                    onLoginError = {}
            )
        }
    }


}

@Composable
fun ButtonWithTextAndIcon(
        modifier: Modifier = Modifier,
        text: String,
        drawableResId: Int,
        iconSize: Int = 24,
        iconColor: Color = Color.Black,
        onClick: () -> Unit

) {

    Row(
            modifier = Modifier.clickable { onClick() },
            verticalAlignment = Alignment.CenterVertically
    ) {
        Text(text = text, color = colorResource(id = R.color.black))
        Spacer(modifier = Modifier.width(8.dp))
        Icon(
                painter = painterResource(id = drawableResId),
                contentDescription = null,
                modifier = Modifier.size(iconSize.dp),
                tint = iconColor
        )
    }
}

@Composable
fun IconButtonWithIcon(
        modifier: Modifier = Modifier,
        iconSize: Int = 24,
        drawableResId: Int,
        iconColor: Color = Color.Black,
        onClick: () -> Unit
) {
    IconButton(
            onClick = onClick,
            modifier = modifier,
    ) {
        Icon(
                painter = painterResource(id = drawableResId),
                contentDescription = "Add",
                modifier = Modifier.size(iconSize.dp),
                tint = iconColor
        )
    }
}

@Composable
fun TabImageButton(modifier: Modifier, text: String, selected: Boolean = false,
                   selectedIcon: Int, unselectedIcon: Int, hidden: Boolean,
                   onClick: () -> Unit) {
    if (!hidden) {


        IconButton(
                onClick = { if (!selected) onClick() },
                modifier = modifier
        ) {
            Column(

                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Top
            ) {
                Image(
                        modifier = Modifier.padding(8.dp),
                        painter = if (!selected) {
                            painterResource(selectedIcon)
                        } else {
                            painterResource(unselectedIcon)
                        },
                        contentDescription = "ic_photo_fixation"
                )
                Text(
                        text = text, modifier = Modifier
                    .padding(vertical = 8.dp, horizontal = 16.dp)
                    .wrapContentSize(),
                        textAlign = TextAlign.Center
                )
            }
        }
    }
}


@Composable
fun ComposableLifecycle(
        lifecycleOwner: LifecycleOwner = LocalLifecycleOwner.current,
        onEvent: (LifecycleOwner, Lifecycle.Event) -> Unit
) {

    DisposableEffect(lifecycleOwner) {
        val observer = LifecycleEventObserver { source, event ->
            onEvent(source, event)
        }
        lifecycleOwner.lifecycle.addObserver(observer)

        onDispose {
            lifecycleOwner.lifecycle.removeObserver(observer)
        }
    }
}
