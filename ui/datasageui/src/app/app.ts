import { BreakpointObserver, Breakpoints } from '@angular/cdk/layout';
import { AsyncPipe } from '@angular/common';
import { Component } from '@angular/core';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatListModule } from '@angular/material/list';
import { MatSidenavModule } from '@angular/material/sidenav';
import { MatToolbarModule } from '@angular/material/toolbar';
import { RouterOutlet } from '@angular/router';
import { map, Observable, shareReplay } from 'rxjs';

@Component({
	selector: 'app-root',
	imports: [RouterOutlet, MatToolbarModule,
		MatButtonModule,
		MatSidenavModule,
		MatListModule,
		MatIconModule, AsyncPipe],
	templateUrl: './app.html',
	styleUrl: './app.css'
})
export class App {
	protected title = 'datasageui';
	isHandset$: Observable<boolean>;
	constructor(private breakpointObserver: BreakpointObserver) {
		this.isHandset$ = this.breakpointObserver.observe([Breakpoints.Handset])
			.pipe(
				map(result => result.matches),
				shareReplay()
			);
	}
}
