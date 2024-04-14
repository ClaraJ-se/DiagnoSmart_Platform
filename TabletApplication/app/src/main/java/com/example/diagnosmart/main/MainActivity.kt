package com.example.diagnosmart.main

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.tooling.preview.Preview
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import com.example.diagnosmart.navigation.DiagnoSmartNavGraph
import com.example.diagnosmart.ui.theme.SymtomSavvyTheme
import com.example.symtomsavvy.R
import com.google.accompanist.systemuicontroller.rememberSystemUiController
import dagger.hilt.android.AndroidEntryPoint


@AndroidEntryPoint
class MainActivity : ComponentActivity() {

    private val diagnoSmartViewModel: DiagnoSmartViewModel by viewModels()


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            SymtomSavvyTheme {
                val uiState by diagnoSmartViewModel.mainState.collectAsState()
                val launchNavigation = remember { mutableStateOf(false) }


                val systemUiController = rememberSystemUiController()
                val useDarkIcons = !isSystemInDarkTheme()

                SplashScreen(true)

                DisposableEffect(key1 = lifecycle, systemUiController, useDarkIcons) {
                    val observer = LifecycleEventObserver { source, event ->
                        when (event) {
                            Lifecycle.Event.ON_RESUME -> {
                                systemUiController.setStatusBarColor(color = Color(20, 115, 138))
                                systemUiController.navigationBarDarkContentEnabled = true
                                onDispose {}
                            }

                            Lifecycle.Event.ON_PAUSE -> {
                                systemUiController.setStatusBarColor(color = Color(20, 115, 138))
                                systemUiController.navigationBarDarkContentEnabled = true
                                onDispose {}
                            }

                            else -> {}
                        }
                    }
                    lifecycle.addObserver(observer)
                    onDispose { lifecycle.removeObserver(observer) }
                }



                LaunchedEffect(uiState) {
                    if (uiState.isSplashFinished) {
                        launchNavigation.value = true

                    }
                }

                if (launchNavigation.value) {
                    SplashScreen(false)
                    DiagnoSmartNavGraph(diagnoSmartViewModel = diagnoSmartViewModel)
                }
            }
        }
    }
}

@Composable
fun SplashScreen(show: Boolean) {
    if (show) {
        Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(Color.White), // Background color of the splash screen
                contentAlignment = Alignment.Center
        ) {
            Image(
                    painter = painterResource(
                            id = R.drawable.splash_img
                    ), // Your splash screen image resource
                    contentDescription = "Splash Image",
                    modifier = Modifier
                        .fillMaxSize(),
                    contentScale = ContentScale.FillBounds
            )
        }
    } else {

        Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(Color.White), // Background color of the splash screen
                contentAlignment = Alignment.Center
        ) {}
    }
}

@Composable
fun Greeting(name: String, modifier: Modifier = Modifier) {
//    Text(
//            text = "Hello $name!",
//            modifier = modifier,
//            color = Color.Black
//
//    )
}

@Preview(showBackground = true)
@Composable
fun GreetingPreview() {
    SymtomSavvyTheme {
        Greeting("Android")
    }
}