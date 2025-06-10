import { Routes } from '@angular/router';
import { ChatComponent } from './components/chat-component/chat-component';
import { Home } from './components/home/home';
import { Document } from './components/document/document';
import { Masterchat } from './components/masterchat/masterchat';

export const routes: Routes = [
	{
		path: '',
		component: Home
	},
	{
		path: 'documents',
		component: Document
	},
	{
		path: 'chat',
		component: ChatComponent
	},
	{
		path: 'masterchat',
		component: Masterchat
	}
];
